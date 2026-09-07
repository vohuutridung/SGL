from __future__ import annotations

import pytest

from sgl.data import (
    ANSWER_PREFIX,
    DEFAULT_SYSTEM_PROMPT,
    IGNORE_INDEX,
    THINK_PREFIX,
    OverlengthSampleError,
    SampleFormatError,
    SpectralDataCollator,
    format_s1k_example,
    native_assistant_end_token_id,
    normalize_final_answer,
    positions_to_ranges,
    prepare_sample,
    split_reasoning_steps,
)


class FakeTokenizer:
    chat_template = "fake-native-template"
    is_fast = True
    eos_token_id = 2
    eos_token = "¤"
    pad_token_id = 0
    pad_token = "PAD"
    padding_side = "right"

    @staticmethod
    def _render(messages, add_generation_prompt=False):
        rendered = "".join(
            f"<{message['role']}>{message['content']}¤" for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    @staticmethod
    def _encode(text):
        return [2 if char == "¤" else ord(char) + 10 for char in text]

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_tensors=None,
    ):
        del return_tensors
        rendered = self._render(messages, add_generation_prompt)
        return self._encode(rendered) if tokenize else rendered

    def __call__(
        self,
        text,
        *,
        add_special_tokens,
        return_offsets_mapping,
        truncation,
    ):
        assert not add_special_tokens
        assert return_offsets_mapping
        assert not truncation
        return {
            "input_ids": self._encode(text),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def sample_row():
    return {
        "question": "What is 1+1?",
        "deepseek_thinking_trajectory": "first\n\nsecond\n",
        "deepseek_attempt": "The answer is \\boxed{2}.",
    }


def test_split_assigns_separator_to_previous_step():
    content = "first\n\nsecond"
    parsed = split_reasoning_steps(content)

    assert len(parsed) == 2
    first, second = parsed
    assert content[first.start : first.end] == "first\n\n"
    assert content[second.start : second.end] == "second"


def test_split_coalesces_consecutive_separators_into_previous_step():
    content = "first\n\n\n\nsecond"
    first, second = split_reasoning_steps(content)
    assert content[first.start : first.end] == "first\n\n\n\n"
    assert content[second.start : second.end] == "second"


def test_s1k_formatter_matches_official_qwen_recipe():
    prompt, target, parsed = format_s1k_example(sample_row())

    assert prompt == [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": "What is 1+1?"},
    ]
    assert target == {
        "role": "assistant",
        "content": (
            f"{THINK_PREFIX}first\n\nsecond"
            f"{ANSWER_PREFIX}Answer: The answer is \\boxed{{2}}."
        ),
    }
    first = parsed.reasoning_steps[0]
    assert target["content"][first.start : first.end] == f"{THINK_PREFIX}first\n\n"
    assert target["content"][parsed.final_start :] == (
        f"{ANSWER_PREFIX}Answer: The answer is \\boxed{{2}}."
    )


def test_final_answer_prefix_is_not_duplicated():
    assert normalize_final_answer("Already says Answer: 2") == "Already says Answer: 2"
    assert normalize_final_answer("Final is 2") == "Answer: Final is 2"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("question", None),
        ("deepseek_thinking_trajectory", ["not", "a", "string"]),
        ("deepseek_attempt", "  "),
    ],
)
def test_s1k_formatter_rejects_invalid_required_fields(field, value):
    row = sample_row()
    row[field] = value
    with pytest.raises(SampleFormatError, match=field):
        format_s1k_example(row)


def test_s1k_formatter_rejects_missing_required_field():
    row = sample_row()
    del row["deepseek_attempt"]
    with pytest.raises(SampleFormatError, match="deepseek_attempt"):
        format_s1k_example(row)


def test_prepare_sample_masks_prompt_and_always_trains_final_and_eos():
    tokenizer = FakeTokenizer()
    prepared = prepare_sample(sample_row(), tokenizer, max_length=32_768)

    assert len(prepared.step_token_positions) == 2
    assert len(prepared.final_answer_positions) > 0
    assert len(prepared.eos_positions) == 1
    separator_positions = prepared.step_token_positions[0][-2:]
    rendered_chars = [
        chr(prepared.input_ids[position] - 10) for position in separator_positions
    ]
    assert rendered_chars == ["\n", "\n"]

    active_ranges = prepared.active_ranges([0])
    labels = prepared.build_labels(active_ranges)
    assert all(labels[position] != IGNORE_INDEX for position in prepared.final_answer_positions)
    assert all(labels[position] != IGNORE_INDEX for position in prepared.eos_positions)
    assert all(
        labels[position] == IGNORE_INDEX
        for position in prepared.step_token_positions[1]
    )
    first_target_position = prepared.step_token_positions[0][0]
    assert all(label == IGNORE_INDEX for label in labels[:first_target_position])


def test_overlength_sample_is_rejected_if_final_answer_would_be_lost():
    tokenizer = FakeTokenizer()
    with pytest.raises(OverlengthSampleError):
        prepare_sample(sample_row(), tokenizer, max_length=20)


def test_truncation_keeps_maximal_step_prefix_and_complete_final_answer():
    tokenizer = FakeTokenizer()
    row = {
        "question": "Q",
        "deepseek_thinking_trajectory": (
            "a" * 20 + "\n\n" + "b" * 20 + "\n\n" + "c" * 20
        ),
        "deepseek_attempt": "FINAL",
    }
    full = prepare_sample(row, tokenizer, max_length=1_000)
    truncated = prepare_sample(row, tokenizer, max_length=len(full.input_ids) - 10)

    assert truncated.truncated
    assert len(truncated.input_ids) <= len(full.input_ids) - 10
    assert len(truncated.step_token_positions) == 2
    final_text = "".join(
        chr(truncated.input_ids[position] - 10)
        for position in truncated.final_answer_positions
    )
    assert final_text == f"{ANSWER_PREFIX}Answer: FINAL"
    assert len(truncated.eos_positions) == 1


def test_native_assistant_terminator_is_derived_from_template():
    assert native_assistant_end_token_id(FakeTokenizer()) == 2


def test_positions_to_ranges():
    assert positions_to_ranges([7, 3, 4, 4, 9]) == [[3, 5], [7, 8], [9, 10]]


def test_collator_right_pads_labels_with_ignore_index():
    tokenizer = FakeTokenizer()
    collator = SpectralDataCollator(tokenizer, pad_to_multiple_of=4)
    batch = collator(
        [
            {"input_ids": [1, 2, 3], "labels": [IGNORE_INDEX, 2, 3]},
            {"input_ids": [4, 5], "labels": [IGNORE_INDEX, 5]},
        ]
    )

    assert batch["input_ids"].shape == (2, 4)
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0], [1, 1, 0, 0]]
    assert batch["labels"].tolist() == [
        [IGNORE_INDEX, 2, 3, IGNORE_INDEX],
        [IGNORE_INDEX, 5, IGNORE_INDEX, IGNORE_INDEX],
    ]
