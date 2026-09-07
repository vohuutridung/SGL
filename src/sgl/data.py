from __future__ import annotations

import hashlib
import random
from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

DEFAULT_DATASET = "simplescaling/s1K-1.1"
DEFAULT_DATASET_REVISION = "96c411f1fe4c49d20f0e2a1565f61e1a28b0b84d"
DEFAULT_SPLIT = "train"
DEFAULT_MAX_SAMPLES = 1_000
DEFAULT_MAX_LENGTH = 32_768
DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
DATASET_FORMAT = "simplescaling_s1k_1_1"
QUESTION_FIELD = "question"
REASONING_FIELD = "deepseek_thinking_trajectory"
ANSWER_FIELD = "deepseek_attempt"
THINK_PREFIX = "<|im_start|>think\n"
ANSWER_PREFIX = "\n<|im_start|>answer\n"
ANSWER_LABEL = "Answer: "
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
) -> tuple[CharStep, ...]:
    if not content:
        raise SampleFormatError("Reasoning trajectory is empty")
    if not separator:
        raise ValueError("separator must not be empty")

    reasoning_steps: list[CharStep] = []
    cursor = 0
    step_id = 0
    while cursor < len(content):
        separator_start = content.find(separator, cursor)
        end = len(content) if separator_start < 0 else separator_start + len(separator)
        while end < len(content) and content.startswith(separator, end):
            end += len(separator)
        if end <= cursor:
            raise RuntimeError("Step parser failed to make progress")
        reasoning_steps.append(CharStep(step_id=step_id, start=cursor, end=end))
        cursor = end
        step_id += 1

    if not reasoning_steps:
        raise SampleFormatError("Reasoning trajectory contains no steps")

    return tuple(reasoning_steps)


def _require_nonempty_string(
    row: Mapping[str, Any],
    field: str,
    *,
    strip: bool = False,
) -> str:
    if field not in row:
        raise SampleFormatError(f"Dataset row is missing required field {field!r}")
    value = row[field]
    if not isinstance(value, str):
        raise SampleFormatError(f"row[{field!r}] must be a string")
    if not value.strip():
        raise SampleFormatError(f"row[{field!r}] must not be empty")
    return value.strip() if strip else value


def normalize_final_answer(answer: str) -> str:
    answer = answer.strip()
    return answer if "Answer:" in answer else ANSWER_LABEL + answer


def format_s1k_example(
    row: Mapping[str, Any],
    *,
    separator: str = "\n\n",
) -> tuple[list[dict[str, str]], dict[str, str], ParsedTarget]:
    """Map one raw s1K-1.1 row to the official Qwen reasoning format."""

    question = _require_nonempty_string(row, QUESTION_FIELD)
    reasoning = _require_nonempty_string(row, REASONING_FIELD, strip=True)
    answer = _require_nonempty_string(row, ANSWER_FIELD, strip=True)
    raw_steps = split_reasoning_steps(reasoning, separator=separator)

    content = THINK_PREFIX + reasoning + ANSWER_PREFIX + normalize_final_answer(answer)
    reasoning_offset = len(THINK_PREFIX)
    reasoning_steps = tuple(
        CharStep(
            step_id=step.step_id,
            start=0 if step.step_id == 0 else reasoning_offset + step.start,
            end=reasoning_offset + step.end,
        )
        for step in raw_steps
    )
    final_start = reasoning_offset + len(reasoning)
    parsed = ParsedTarget(
        content=content,
        reasoning_steps=reasoning_steps,
        final_start=final_start,
        final_end=len(content),
    )
    prompt_messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    target_message = {"role": "assistant", "content": content}
    return prompt_messages, target_message, parsed


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
) -> PreparedSample:
    if max_length < 2:
        raise ValueError("max_length must be at least 2")

    prompt, target, parsed = format_s1k_example(row, separator=separator)
    rendered, input_ids, offsets = _render_with_native_template(
        tokenizer,
        [*prompt, target],
    )
    if len(input_ids) > max_length:
        # Preserve the prompt and complete final answer, then keep the longest
        # prefix of whole reasoning steps that fits.
        reasoning = _require_nonempty_string(row, REASONING_FIELD, strip=True)
        raw_steps = split_reasoning_steps(reasoning, separator=separator)
        for kept_steps in range(len(parsed.reasoning_steps) - 1, 0, -1):
            prefix_end = raw_steps[kept_steps - 1].end
            reasoning_prefix = reasoning[:prefix_end]
            if reasoning_prefix.endswith(separator):
                reasoning_prefix = reasoning_prefix[: -len(separator)]
            truncated_row = {
                QUESTION_FIELD: row[QUESTION_FIELD],
                REASONING_FIELD: reasoning_prefix,
                ANSWER_FIELD: row[ANSWER_FIELD],
            }
            truncated_prompt, truncated_target, _ = format_s1k_example(
                truncated_row,
                separator=separator,
            )
            _, candidate_ids, _ = _render_with_native_template(
                tokenizer,
                [*truncated_prompt, truncated_target],
            )
            if len(candidate_ids) > max_length:
                continue
            result = prepare_sample(
                truncated_row,
                tokenizer,
                max_length=max_length,
                separator=separator,
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
    if not final_positions:
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
