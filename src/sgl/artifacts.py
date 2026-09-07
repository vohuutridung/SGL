from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
MANIFEST_FILENAME = "manifest.json"
MASKS_FILENAME = "masks.jsonl"
SHARDS_DIRNAME = "shards"


@dataclass(frozen=True)
class MaskRecord:
    sample_position: int
    source_index: int
    input_ids_hash: str
    sequence_length: int
    truncated: bool
    spectral_rank: int
    rank_cumulative_energy: float
    step_strengths: tuple[float, ...]
    selected_step_ids: tuple[int, ...]
    active_token_ranges: tuple[tuple[int, int], ...]
    reasoning_tokens: int
    selected_reasoning_tokens: int
    final_answer_tokens: int
    eos_tokens: int

    @property
    def token_retention_ratio(self) -> float:
        if self.reasoning_tokens == 0:
            return 0.0
        return self.selected_reasoning_tokens / self.reasoning_tokens

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MaskRecord:
        values = dict(payload)
        values["step_strengths"] = tuple(float(value) for value in values["step_strengths"])
        values["selected_step_ids"] = tuple(
            int(value) for value in values["selected_step_ids"]
        )
        values["active_token_ranges"] = tuple(
            (int(start), int(end)) for start, end in values["active_token_ranges"]
        )
        return cls(**values)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint_local_model(path: str | Path) -> str | None:
    root = Path(path)
    if not root.is_dir():
        return None
    candidates = sorted(
        file
        for file in root.rglob("*")
        if file.is_file()
        and (
            file.suffix
            in {
                ".safetensors",
                ".bin",
                ".json",
                ".py",
                ".model",
                ".txt",
                ".tiktoken",
                ".jinja",
                ".yaml",
                ".yml",
            }
            or file.name.startswith(("tokenizer", "vocab", "merges"))
        )
    )
    if not candidates:
        raise ValueError(f"Local model directory contains no fingerprintable files: {root}")

    digest = hashlib.sha256()
    for file in candidates:
        relative = file.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(str(file.stat().st_size).encode())
        with file.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, destination)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def load_mask_records(path: str | Path) -> list[MaskRecord]:
    records = [MaskRecord.from_dict(payload) for payload in iter_jsonl(path)]
    positions = [record.sample_position for record in records]
    if positions != sorted(positions):
        raise ValueError("Mask records must be sorted by sample_position")
    if len(positions) != len(set(positions)):
        raise ValueError("Mask records contain duplicate sample_position values")
    return records


def completed_source_indices(path: str | Path) -> set[int]:
    source_indices: set[int] = set()
    destination = Path(path)
    if not destination.exists():
        return source_indices
    for payload in iter_jsonl(destination):
        source_indices.add(int(payload["source_index"]))
    return source_indices


def shard_path(output_dir: str | Path, process_index: int) -> Path:
    return (
        Path(output_dir)
        / SHARDS_DIRNAME
        / f"rank-{process_index:05d}.jsonl"
    )


def merge_mask_shards(
    output_dir: str | Path,
    *,
    num_processes: int,
) -> list[MaskRecord]:
    output_path = Path(output_dir)
    records: list[MaskRecord] = []
    for process_index in range(num_processes):
        path = shard_path(output_path, process_index)
        if not path.exists():
            raise FileNotFoundError(f"Missing mask shard: {path}")
        records.extend(MaskRecord.from_dict(payload) for payload in iter_jsonl(path))

    records.sort(key=lambda record: record.sample_position)
    sample_positions = [record.sample_position for record in records]
    if len(sample_positions) != len(set(sample_positions)):
        raise ValueError("Mask shards contain duplicate sample positions")

    destination = output_path / MASKS_FILENAME
    temporary = destination.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True, ensure_ascii=False))
            handle.write("\n")
    os.replace(temporary, destination)
    return records


def summarize_records(records: Iterable[MaskRecord]) -> dict[str, float | int]:
    records = list(records)
    if not records:
        return {
            "num_samples": 0,
            "reasoning_tokens": 0,
            "selected_reasoning_tokens": 0,
            "reasoning_token_retention": 0.0,
            "mean_spectral_rank": 0.0,
        }

    reasoning_tokens = sum(record.reasoning_tokens for record in records)
    selected_tokens = sum(record.selected_reasoning_tokens for record in records)
    return {
        "num_samples": len(records),
        "reasoning_tokens": reasoning_tokens,
        "selected_reasoning_tokens": selected_tokens,
        "reasoning_token_retention": (
            selected_tokens / reasoning_tokens if reasoning_tokens else 0.0
        ),
        "mean_spectral_rank": (
            sum(record.spectral_rank for record in records) / len(records)
        ),
        "final_answer_tokens": sum(record.final_answer_tokens for record in records),
        "eos_tokens": sum(record.eos_tokens for record in records),
        "truncated_samples": sum(record.truncated for record in records),
    }
