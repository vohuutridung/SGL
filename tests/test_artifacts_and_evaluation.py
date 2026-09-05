from __future__ import annotations

from types import SimpleNamespace

from sgl.artifacts import (
    MaskRecord,
    append_jsonl,
    fingerprint_local_model,
    load_mask_records,
    merge_mask_shards,
    shard_path,
    summarize_records,
)
from sgl.evaluation import generation_eos_token_ids, grade_response


def _record(sample_position: int, source_index: int) -> MaskRecord:
    return MaskRecord(
        sample_position=sample_position,
        source_index=source_index,
        input_ids_hash=f"hash-{source_index}",
        sequence_length=10,
        truncated=False,
        spectral_rank=2,
        rank_cumulative_energy=0.96,
        step_strengths=(0.7, 0.3),
        selected_step_ids=(0,),
        active_token_ranges=((3, 6), (8, 10)),
        reasoning_tokens=6,
        selected_reasoning_tokens=4,
        final_answer_tokens=1,
        eos_tokens=1,
    )


def test_mask_shards_are_merged_in_sample_order(tmp_path):
    append_jsonl(shard_path(tmp_path, 0), _record(2, 12).to_dict())
    append_jsonl(shard_path(tmp_path, 0), _record(0, 10).to_dict())
    append_jsonl(shard_path(tmp_path, 1), _record(1, 11).to_dict())

    merged = merge_mask_shards(tmp_path, num_processes=2)
    loaded = load_mask_records(tmp_path / "masks.jsonl")

    assert [record.sample_position for record in merged] == [0, 1, 2]
    assert loaded == merged
    summary = summarize_records(loaded)
    assert summary["num_samples"] == 3
    assert summary["reasoning_token_retention"] == 2 / 3


def test_local_model_fingerprint_changes_with_weights(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"test"}', encoding="utf-8")
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"first")
    first = fingerprint_local_model(tmp_path)
    weights.write_bytes(b"second")
    assert fingerprint_local_model(tmp_path) != first


def test_math_verify_grades_numeric_and_latex_answers():
    correct, _ = grade_response(
        "After checking, the final answer is $\\boxed{18}$.",
        "18",
        gold_is_latex=False,
    )
    assert correct

    correct, _ = grade_response(
        "An intermediate value is $\\boxed{3}$, but the final answer is $4$.",
        "4",
        gold_is_latex=False,
    )
    assert correct

    correct, _ = grade_response(
        "The final cost is $\\boxed{\\$32,348}$.",
        "\\$32,\\!348",
        gold_is_latex=True,
    )
    assert correct


def test_generation_uses_native_turn_terminator_and_model_eos(monkeypatch):
    monkeypatch.setattr("sgl.evaluation.native_assistant_end_token_id", lambda _: 17)
    model = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=11),
        config=SimpleNamespace(eos_token_id=11),
    )
    tokenizer = SimpleNamespace(eos_token_id=13)
    assert generation_eos_token_ids(model, tokenizer) == [11, 13, 17]

    correct, _ = grade_response(
        "Therefore $\\boxed{\\left(3, \\frac{\\pi}{2}\\right)}$.",
        "\\left(3, \\frac{\\pi}{2}\\right)",
        gold_is_latex=True,
    )
    assert correct
