from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from sgl.data import PreparedSample


@dataclass(frozen=True)
class SpectralSelection:
    rank: int
    cumulative_energy: float
    singular_values: tuple[float, ...]
    step_strengths: tuple[float, ...]
    selected_step_ids: tuple[int, ...]
    reasoning_tokens: int
    selected_reasoning_tokens: int

    @property
    def token_retention_ratio(self) -> float:
        if self.reasoning_tokens == 0:
            return 0.0
        return self.selected_reasoning_tokens / self.reasoning_tokens


def _validate_threshold(name: str, value: float) -> None:
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value}")


def exact_spectral_selection(
    gradient_matrix: torch.Tensor,
    step_ids: torch.Tensor,
    *,
    num_steps: int,
    rank_threshold: float = 0.95,
    selection_threshold: float = 0.8,
    svd_driver: str | None = "gesvd",
) -> SpectralSelection:
    """Apply equations (3), (4), (7), and (8) to one sample.

    This intentionally computes a full reduced SVD, not a randomized or
    truncated approximation. ``gradient_matrix`` is always promoted to FP32.
    """

    _validate_threshold("rank_threshold", rank_threshold)
    _validate_threshold("selection_threshold", selection_threshold)
    if gradient_matrix.ndim != 2:
        raise ValueError("gradient_matrix must have shape [tokens, hidden_size]")
    if step_ids.ndim != 1 or step_ids.shape[0] != gradient_matrix.shape[0]:
        raise ValueError("step_ids must align one-to-one with gradient rows")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if gradient_matrix.shape[0] == 0:
        raise ValueError("gradient_matrix has no token rows")

    matrix = gradient_matrix.to(dtype=torch.float32)
    step_ids = step_ids.to(device=matrix.device, dtype=torch.long)
    if int(step_ids.min()) < 0 or int(step_ids.max()) >= num_steps:
        raise ValueError("step_ids contain an out-of-range reasoning step")
    present_steps = torch.bincount(step_ids, minlength=num_steps)
    if torch.any(present_steps == 0):
        missing = torch.nonzero(present_steps == 0).flatten().tolist()
        raise ValueError(f"Reasoning steps without gradient rows: {missing}")

    svd_kwargs: dict[str, Any] = {"full_matrices": False}
    if matrix.device.type == "cuda" and svd_driver is not None:
        svd_kwargs["driver"] = svd_driver
    u, singular_values, _ = torch.linalg.svd(matrix, **svd_kwargs)

    squared = singular_values.square()
    total_energy = squared.sum()
    if not torch.isfinite(total_energy) or float(total_energy) <= 0.0:
        raise ValueError("Gradient matrix has zero or non-finite spectral energy")
    cumulative = squared.cumsum(dim=0) / total_energy
    threshold = torch.tensor(rank_threshold, device=cumulative.device, dtype=cumulative.dtype)
    rank = int(torch.searchsorted(cumulative, threshold, right=False).item()) + 1
    rank = min(rank, singular_values.numel())

    # Equation (7): unweighted truncated leverage score. Do not multiply by
    # sigma^2; that would be projection energy and would change the method.
    token_leverage = u[:, :rank].square().sum(dim=1)
    strengths: list[float] = []
    for step_id in range(num_steps):
        score = token_leverage[step_ids == step_id].mean()
        strengths.append(float(score.detach().cpu()))

    total_strength = sum(strengths)
    if not total_strength > 0.0:
        raise ValueError("Step spectral strengths sum to zero")

    # Python's sort is stable. The explicit original index makes the requested
    # tie behavior clear and independent of input container ordering.
    ranked_steps = sorted(range(num_steps), key=lambda index: (-strengths[index], index))
    selected: list[int] = []
    selected_strength = 0.0
    for step_id in ranked_steps:
        selected.append(step_id)
        selected_strength += strengths[step_id]
        if selected_strength / total_strength >= selection_threshold:
            break

    selected_set = set(selected)
    selected_tokens = sum(
        int(count)
        for step_id, count in enumerate(present_steps.detach().cpu().tolist())
        if step_id in selected_set
    )
    return SpectralSelection(
        rank=rank,
        cumulative_energy=float(cumulative[rank - 1].detach().cpu()),
        singular_values=tuple(float(value) for value in singular_values.detach().cpu()),
        step_strengths=tuple(strengths),
        selected_step_ids=tuple(selected),
        reasoning_tokens=int(gradient_matrix.shape[0]),
        selected_reasoning_tokens=selected_tokens,
    )


def get_decoder(model: Any) -> Any:
    if hasattr(model, "get_decoder"):
        try:
            decoder = model.get_decoder()
        except (AttributeError, NotImplementedError):
            decoder = None
        if decoder is not None:
            return decoder

    base_model_prefix = getattr(model, "base_model_prefix", "")
    if base_model_prefix and hasattr(model, base_model_prefix):
        return getattr(model, base_model_prefix)
    if hasattr(model, "model"):
        return model.model
    raise TypeError("Could not locate the causal LM decoder")


def capture_pre_lm_head_hidden(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the final normalized hidden state immediately before lm_head."""

    decoder = get_decoder(model)
    with torch.no_grad():
        outputs = decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
    if hidden.ndim != 3:
        raise ValueError("Decoder must return hidden states with shape [batch, sequence, hidden]")
    return hidden


def capture_reasoning_gradient_matrix(
    model: Any,
    prepared: PreparedSample,
    *,
    device: torch.device,
    lm_head_chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture g_t = d ell_t / d h_t for all reasoning tokens in one sample.

    The hidden tensor is the decoder output immediately before the LM head.
    For a causal LM, hidden position ``t - 1`` predicts target token ``t``.
    Final-answer and EOS tokens are deliberately excluded from the SVD; they
    are always supervised later.
    """

    if lm_head_chunk_size <= 0:
        raise ValueError("lm_head_chunk_size must be positive")

    label_positions = [
        position
        for position, step_id in enumerate(prepared.reasoning_step_ids)
        if step_id >= 0
    ]
    if not label_positions:
        raise ValueError("Prepared sample contains no reasoning tokens")
    if label_positions[0] == 0:
        raise ValueError("Causal target position zero has no predictor hidden state")

    input_ids = torch.tensor(prepared.input_ids, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    hidden = capture_pre_lm_head_hidden(model, input_ids, attention_mask)

    label_positions_tensor = torch.tensor(label_positions, dtype=torch.long, device=device)
    predictor_positions = label_positions_tensor - 1
    hidden_rows = hidden[0].index_select(0, predictor_positions)
    target_ids = input_ids[0].index_select(0, label_positions_tensor)
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise TypeError("Model has no output embedding / LM head")

    gradient_chunks: list[torch.Tensor] = []
    for start in range(0, len(label_positions), lm_head_chunk_size):
        end = min(start + lm_head_chunk_size, len(label_positions))
        leaf = hidden_rows[start:end].detach().requires_grad_(True)
        logits = lm_head(leaf)
        loss = F.cross_entropy(
            logits.float(),
            target_ids[start:end],
            reduction="sum",
        )
        (gradient,) = torch.autograd.grad(loss, leaf, retain_graph=False, create_graph=False)
        gradient_chunks.append(gradient.detach().to(device="cpu", dtype=torch.float32))

    gradient_matrix = torch.cat(gradient_chunks, dim=0)
    step_ids = torch.tensor(
        [prepared.reasoning_step_ids[position] for position in label_positions],
        dtype=torch.long,
    )
    if gradient_matrix.shape[0] != step_ids.numel():
        raise RuntimeError("Captured gradients and step IDs are misaligned")
    return gradient_matrix, step_ids
