from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from sgl.data import PreparedSample
from sgl.spectral import (
    capture_reasoning_gradient_matrix,
    exact_spectral_selection,
)


def test_exact_svd_rank_leverage_and_stable_tie_selection():
    gradients = torch.diag(torch.tensor([4.0, 2.0, 1.0]))
    step_ids = torch.tensor([0, 1, 2])

    result = exact_spectral_selection(
        gradients,
        step_ids,
        num_steps=3,
        rank_threshold=0.8,
        selection_threshold=0.8,
        svd_driver=None,
    )

    # 16/21 < .8 and 20/21 >= .8.
    assert result.rank == 2
    assert result.cumulative_energy >= 0.8
    assert result.step_strengths == (1.0, 1.0, 0.0)
    # Steps 0 and 1 tie, so their original order must be preserved.
    assert result.selected_step_ids == (0, 1)
    assert sum(result.step_strengths) == result.rank


def test_spectral_strength_is_unweighted_leverage_not_projection_energy():
    theta = torch.tensor(0.45)
    u = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta), torch.cos(theta)],
        ]
    )
    singular = torch.diag(torch.tensor([10.0, 1.0]))
    gradients = u @ singular

    result = exact_spectral_selection(
        gradients,
        torch.tensor([0, 1]),
        num_steps=2,
        rank_threshold=0.5,
        selection_threshold=0.5,
        svd_driver=None,
    )

    assert result.rank == 1
    expected = (float(torch.cos(theta).square()), float(torch.sin(theta).square()))
    assert torch.allclose(
        torch.tensor(result.step_strengths),
        torch.tensor(expected),
        atol=1e-6,
    )


class DummyDecoder(nn.Module):
    def __init__(self, hidden_table: torch.Tensor):
        super().__init__()
        self.register_buffer("hidden_table", hidden_table)

    def forward(self, input_ids, attention_mask, use_cache, return_dict):
        del attention_mask, use_cache, return_dict
        return SimpleNamespace(last_hidden_state=self.hidden_table[input_ids])


class DummyCausalLM(nn.Module):
    base_model_prefix = "decoder"

    def __init__(self):
        super().__init__()
        hidden_table = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
                [0.5, -0.5],
            ]
        )
        self.decoder = DummyDecoder(hidden_table)
        self.lm_head = nn.Linear(2, 5, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0],
                        [-1.0, 0.0],
                        [0.0, -1.0],
                    ]
                )
            )

    def get_decoder(self):
        return self.decoder

    def get_output_embeddings(self):
        return self.lm_head


def test_gradient_capture_uses_causal_shift_and_pre_lm_head_gradient():
    model = DummyCausalLM()
    prepared = PreparedSample(
        input_ids=[0, 1, 2, 3],
        reasoning_step_ids=[-1, 0, 0, -1],
        final_answer_mask=[False, False, False, True],
        eos_mask=[False, False, False, False],
        step_char_spans=(),
        step_token_positions=((1, 2),),
        final_answer_positions=(3,),
        eos_positions=(),
        truncated=False,
    )

    gradients, step_ids = capture_reasoning_gradient_matrix(
        model,
        prepared,
        device=torch.device("cpu"),
        lm_head_chunk_size=1,
    )

    expected = []
    for predictor_position, target in [(0, 1), (1, 2)]:
        hidden = model.decoder.hidden_table[prepared.input_ids[predictor_position]]
        logits = model.lm_head(hidden)
        probabilities = logits.softmax(dim=-1)
        one_hot = torch.nn.functional.one_hot(
            torch.tensor(target), num_classes=logits.numel()
        ).float()
        expected.append(model.lm_head.weight.T @ (probabilities - one_hot))

    assert torch.allclose(gradients, torch.stack(expected), atol=1e-6)
    assert step_ids.tolist() == [0, 0]
    assert gradients.dtype == torch.float32
