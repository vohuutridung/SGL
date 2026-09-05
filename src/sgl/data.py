from __future__ import annotations

import hashlib
import random
from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

DEFAULT_DATASET = "Elliott/Openr1-Math-46k-8192"
DEFAULT_DATASET_REVISION = "bd025079093aec0409f535a5b2cae1c28019d917"
DEFAULT_SPLIT = "train"
DEFAULT_MAX_LENGTH = 32_768
IGNORE_INDEX = -100


class SampleFormatError(ValueError):
    """Raised when a dataset row cannot satisfy the configured SGL format."""


class OverlengthSampleError(SampleFormatError):
    """Raised when boundary truncation would remove the always-trained final answer."""


@dataclass(frozen=True)
class CharStep:
    step_id: int
    start: int
    end: int


@dataclass(frozen=True)
class ParsedTarget:
    content: str
    reasoning_steps: tuple[CharStep, ...]
    final_start: int
    final_end: int


@dataclass
class PreparedSample:
    input_ids: list[int]
    reasoning_step_ids: list[int]
    final_answer_mask: list[bool]
    eos_mask: list[bool]
    step_char_spans: tuple[CharStep, ...]
    step_token_positions: tuple[tuple[int, ...], ...]
    final_answer_positions: tuple[int, ...]
    eos_positions: tuple[int, ...]
    truncated: bool

    def __post_init__(self) -> None:
        length = len(self.input_ids)
        if not (
            len(self.reasoning_step_ids)
            == len(self.final_answer_mask)
            == len(self.eos_mask)
            == length
        ):
            raise ValueError("All token-aligned fields must have the same length")

    @property
    def input_ids_hash(self) -> str:
        payload = ",".join(str(token_id) for token_id in self.input_ids).encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def reasoning_token_count(self) -> int:
        return sum(step_id >= 0 for step_id in self.reasoning_step_ids)

    def active_positions(self, selected_step_ids: Iterable[int]) -> list[int]:
        selected = set(selected_step_ids)
        positions = [
            position
            for position, step_id in enumerate(self.reasoning_step_ids)
            if step_id in selected
        ]
        positions.extend(self.final_answer_positions)
        positions.extend(self.eos_positions)
        return sorted(set(positions))

    def active_ranges(self, selected_step_ids: Iterable[int]) -> list[list[int]]:
        return positions_to_ranges(self.active_positions(selected_step_ids))

    def build_labels(self, active_ranges: Sequence[Sequence[int]]) -> list[int]:
        labels = [IGNORE_INDEX] * len(self.input_ids)
        for start, end in active_ranges:
            if not (0 <= start <= end <= len(labels)):
                raise ValueError(f"Invalid active token range [{start}, {end})")
            labels[start:end] = self.input_ids[start:end]
        if labels and labels[0] != IGNORE_INDEX:
            raise ValueError("The first causal-LM position cannot be supervised")
        if all(label == IGNORE_INDEX for label in labels):
            raise ValueError("A training sample must supervise at least one token")
        return labels


def positions_to_ranges(positions: Iterable[int]) -> list[list[int]]:
    ordered = sorted(set(positions))
    if not ordered:
        return []

    ranges: list[list[int]] = []
    start = previous = ordered[0]
    for position in ordered[1:]:
        if position == previous + 1:
            previous = position
            continue
        ranges.append([start, previous + 1])
        start = previous = position
    ranges.append([start, previous + 1])
    return ranges


def choose_source_indices(
    total_rows: int,
    max_samples: int | None,
    seed: int,
) -> list[int]:
    if total_rows < 0:
        raise ValueError("total_rows must be non-negative")
    if max_samples is None or max_samples <= 0 or max_samples >= total_rows:
        return list(range(total_rows))
    return random.Random(seed).sample(range(total_rows), max_samples)


def split_reasoning_steps(
    content: str,
    *,
    separator: str = "\n\n",
    final_marker: str = "</think>",
    require_final_answer: bool = True,
) -> ParsedTarget:
    if not content:
        raise SampleFormatError("Assistant target is empty")
    if not separator:
        raise ValueError("separator must not be empty")

    marker_start = content.rfind(final_marker)
    if marker_start < 0:
        raise SampleFormatError(f"Assistant target does not contain {final_marker!r}")

    marker_end = marker_start + len(final_marker)
    final_start = marker_end
    if content.startswith(separator, marker_end):
        # The separator belongs to the preceding reasoning step.
        final_start += len(separator)

    final_text = content[final_start:]
    if require_final_answer and not final_text.strip():
        raise SampleFormatError("Assistant target has no final answer after the thinking block")

    reasoning_steps: list[CharStep] = []
    cursor = 0
    step_id = 0
    while cursor < final_start:
        separator_start = content.find(separator, cursor, final_start)
        end = final_start if separator_start < 0 else separator_start + len(separator)
        if end <= cursor:
            raise RuntimeError("Step parser failed to make progress")
        reasoning_steps.append(CharStep(step_id=step_id, start=cursor, end=end))
        cursor = end
        step_id += 1

    if not reasoning_steps:
        raise SampleFormatError("Assistant target contains no reasoning step")

    return ParsedTarget(
        content=content,
        reasoning_steps=tuple(reasoning_steps),
        final_start=final_start,
        final_end=len(content),
    )


def extract_messages(row: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, str]]:
    prompt = row.get("prompt")
    target = row.get("target")
    if not isinstance(prompt, list) or not prompt:
        raise SampleFormatError("row['prompt'] must be a non-empty message list")
    if not isinstance(target, list) or len(target) != 1:
        raise SampleFormatError("row['target'] must contain exactly one assistant message")

    prompt_messages = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in prompt
    ]
    target_message = {
        "role": str(target[0]["role"]),
        "content": str(target[0]["content"]),
    }
    if target_message["role"] != "assistant":
        raise SampleFormatError("The target message must have role='assistant'")
    return prompt_messages, target_message


def _as_token_id_list(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("Expected a single tokenized sequence")
        value = value[0]
    return [int(token_id) for token_id in value]


def _render_with_native_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
) -> tuple[str, list[int], list[tuple[int, int]]]:
    if not getattr(tokenizer, "chat_template", None):
        raise SampleFormatError("Tokenizer has no native chat template")
    if not getattr(tokenizer, "is_fast", False):
        raise SampleFormatError("A fast tokenizer is required for character-to-token offsets")

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    input_ids = _as_token_id_list(encoded["input_ids"])
    offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]

    native_ids = _as_token_id_list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    if input_ids != native_ids:
        raise SampleFormatError(
            "Rendering then tokenizing does not match native apply_chat_template tokenization"
        )
    return rendered, input_ids, offsets


def _find_content_start(rendered: str, content: str) -> int:
    content_start = rendered.rfind(content)
    if content_start < 0:
        raise SampleFormatError("Native chat template did not preserve assistant content verbatim")
    return content_start


def native_assistant_end_token_id(tokenizer: Any) -> int:
    """Derive the assistant turn terminator emitted by the native template."""

    sentinel = "SGL_ASSISTANT_TERMINATOR_SENTINEL_7D41"
    messages = [
        {"role": "user", "content": "SGL template probe"},
        {"role": "assistant", "content": sentinel},
    ]
    rendered, input_ids, offsets = _render_with_native_template(tokenizer, messages)
    content_start = _find_content_start(rendered, sentinel)
    content_end = content_start + len(sentinel)
    for position, (start, end) in enumerate(offsets):
        if start >= content_end and rendered[start:end].strip():
            return int(input_ids[position])
    raise SampleFormatError("Native chat template has no assistant end-of-turn token")


def _token_anchor(
    offset: tuple[int, int],
    *,
    content_start: int,
    content_end: int,
) -> int | None:
    start, end = offset
    if end <= content_start or start >= content_end:
        return None
    return max(start, content_start) - content_start


def prepare_sample(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    separator: str = "\n\n",
    final_marker: str = "</think>",
    require_final_answer: bool = True,
) -> PreparedSample:
    if max_length < 2:
        raise ValueError("max_length must be at least 2")

    prompt, target = extract_messages(row)
    parsed = split_reasoning_steps(
        target["content"],
        separator=separator,
        final_marker=final_marker,
        require_final_answer=require_final_answer,
    )
    rendered, input_ids, offsets = _render_with_native_template(
        tokenizer,
        [*prompt, target],
    )
    if len(input_ids) > max_length:
        # Preserve the prompt and complete final answer, then keep the longest
        # prefix of whole reasoning steps that fits. If the removed suffix
        # contained </think>, close the retained prefix before reattaching the
        # final answer.
        for kept_steps in range(len(parsed.reasoning_steps) - 1, 0, -1):
            prefix_end = parsed.reasoning_steps[kept_steps - 1].end
            reasoning_prefix = parsed.content[:prefix_end]
            if reasoning_prefix.endswith(separator):
                reasoning_prefix = reasoning_prefix[: -len(separator)]
            truncated_content = (
                reasoning_prefix.rstrip("\n")
                + "\n"
                + final_marker
                + separator
                + parsed.content[parsed.final_start :]
            )
            truncated_target = {"role": "assistant", "content": truncated_content}
            _, candidate_ids, _ = _render_with_native_template(
                tokenizer,
                [*prompt, truncated_target],
            )
            if len(candidate_ids) > max_length:
                continue
            truncated_row = {"prompt": prompt, "target": [truncated_target]}
            result = prepare_sample(
                truncated_row,
                tokenizer,
                max_length=max_length,
                separator=separator,
                final_marker=final_marker,
                require_final_answer=require_final_answer,
            )
            result.truncated = True
            return result
        raise OverlengthSampleError(
            "Prompt, one complete reasoning step, final answer, and end-of-turn "
            f"token do not fit under max_length={max_length}"
        )

    content_start = _find_content_start(rendered, parsed.content)
    content_end = content_start + len(parsed.content)
    step_ends = [step.end for step in parsed.reasoning_steps]

    reasoning_step_ids = [-1] * len(input_ids)
    final_answer_mask = [False] * len(input_ids)
    eos_mask = [False] * len(input_ids)
    step_positions: list[list[int]] = [[] for _ in parsed.reasoning_steps]
    final_positions: list[int] = []

    for position, offset in enumerate(offsets):
        anchor = _token_anchor(
            offset,
            content_start=content_start,
            content_end=content_end,
        )
        if anchor is None:
            continue
        if anchor >= parsed.final_start:
            final_answer_mask[position] = True
            final_positions.append(position)
            continue

        step_id = bisect_right(step_ends, anchor)
        if step_id >= len(parsed.reasoning_steps):
            raise SampleFormatError(f"Could not assign target character {anchor} to a step")
        reasoning_step_ids[position] = step_id
        step_positions[step_id].append(position)

    if any(not positions for positions in step_positions):
        empty = [index for index, positions in enumerate(step_positions) if not positions]
        raise SampleFormatError(f"Reasoning steps without tokens: {empty}")
    if require_final_answer and not final_positions:
        raise SampleFormatError("Final answer contains no tokens")

    last_target_position = max(
        [
            position
            for position, step_id in enumerate(reasoning_step_ids)
            if step_id >= 0
        ]
        + final_positions
    )
    # Native templates may terminate an assistant turn with a dedicated token
    # that differs from tokenizer.eos_token_id (Qwen uses <|im_end|> here while
    # tokenizer.eos_token is <|endoftext|>). Supervise the first non-whitespace
    # template token after the assistant content.
    eos_positions: list[int] = []
    for position in range(last_target_position + 1, len(input_ids)):
        start, end = offsets[position]
        if start < content_end:
            continue
        if rendered[start:end].strip():
            eos_positions.append(position)
            break
    if not eos_positions:
        raise SampleFormatError(
            "Native chat template did not append an assistant end-of-turn token"
        )
    for position in eos_positions:
        eos_mask[position] = True

    if input_ids and (
        reasoning_step_ids[0] >= 0 or final_answer_mask[0] or eos_mask[0]
    ):
        raise SampleFormatError("Target unexpectedly begins at causal position zero")

    return PreparedSample(
        input_ids=input_ids,
        reasoning_step_ids=reasoning_step_ids,
        final_answer_mask=final_answer_mask,
        eos_mask=eos_mask,
        step_char_spans=parsed.reasoning_steps,
        step_token_positions=tuple(tuple(positions) for positions in step_positions),
        final_answer_positions=tuple(final_positions),
        eos_positions=tuple(eos_positions),
        truncated=False,
    )


@dataclass
class SpectralDataCollator:
    tokenizer: Any
    pad_to_multiple_of: int | None = 8

    def __post_init__(self) -> None:
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer needs either pad_token_id or eos_token_id")
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty batch")
        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_length = ((max_length + multiple - 1) // multiple) * multiple

        batch_size = len(features)
        input_ids = torch.full(
            (batch_size, max_length),
            int(self.tokenizer.pad_token_id),
            dtype=torch.long,
        )
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        labels = torch.full(
            (batch_size, max_length),
            IGNORE_INDEX,
            dtype=torch.long,
        )

        for row, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[row, :length] = torch.as_tensor(feature["input_ids"], dtype=torch.long)
            attention_mask[row, :length] = 1
            labels[row, :length] = torch.as_tensor(feature["labels"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
