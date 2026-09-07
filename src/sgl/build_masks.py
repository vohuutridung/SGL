from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
import random
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from sgl.artifacts import (
    MANIFEST_FILENAME,
    SCHEMA_VERSION,
    MaskRecord,
    append_jsonl,
    atomic_write_json,
    completed_source_indices,
    fingerprint_local_model,
    merge_mask_shards,
    shard_path,
    summarize_records,
    utc_now_iso,
)
from sgl.data import (
    ANSWER_FIELD,
    ANSWER_LABEL,
    ANSWER_PREFIX,
    DATASET_FORMAT,
    DEFAULT_DATASET,
    DEFAULT_DATASET_REVISION,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_SPLIT,
    DEFAULT_SYSTEM_PROMPT,
    QUESTION_FIELD,
    REASONING_FIELD,
    THINK_PREFIX,
    SampleFormatError,
    prepare_sample,
)
from sgl.spectral import capture_reasoning_gradient_matrix, exact_spectral_selection

LOGGER = logging.getLogger("sgl.build_masks")
RUN_CONFIG_FILENAME = "run_config.json"
SOURCE_INDICES_FILENAME = "source_indices.json"
PREVALIDATION_FILENAME = "prevalidation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed per-sample Spectral-guided Learning masks."
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Process all 1,000 s1K-1.1 rows by default. Use 0 to process all rows.",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--separator", default="\n\n")
    parser.add_argument("--rank-threshold", type=float, default=0.95)
    parser.add_argument("--selection-threshold", type=float, default=0.8)
    parser.add_argument("--lm-head-chunk-size", type=int, default=256)
    parser.add_argument(
        "--svd-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--svd-driver",
        default="gesvd",
        help="CUDA torch.linalg.svd driver. Ignored on CPU.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _source_indices_hash(indices: list[int]) -> str:
    payload = ",".join(str(index) for index in indices).encode()
    return hashlib.sha256(payload).hexdigest()


def _resolve_svd_device(requested: str, process_device: torch.device) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if process_device.type != "cuda":
            raise RuntimeError("--svd-device=cuda requires a CUDA accelerator process")
        return process_device
    return process_device if process_device.type == "cuda" else torch.device("cpu")


def _resolve_run_revisions(args: argparse.Namespace) -> tuple[str | None, str | None]:
    existing_config = Path(args.output_dir) / RUN_CONFIG_FILENAME
    if existing_config.exists():
        existing = json.loads(existing_config.read_text(encoding="utf-8"))
        return (
            existing.get("resolved_model_revision"),
            existing.get("resolved_dataset_revision"),
        )

    api = HfApi()
    resolved_model_revision = None
    if not Path(args.model_name_or_path).is_dir():
        resolved_model_revision = api.model_info(
            args.model_name_or_path,
            revision=args.model_revision,
        ).sha

    resolved_dataset_revision = None
    if not Path(args.dataset_name).exists():
        resolved_dataset_revision = api.dataset_info(
            args.dataset_name,
            revision=args.dataset_revision,
        ).sha
    return resolved_model_revision, resolved_dataset_revision


def _requested_config(
    args: argparse.Namespace,
    num_processes: int,
    *,
    resolved_model_revision: str | None,
    resolved_dataset_revision: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_name_or_path": args.model_name_or_path,
        "model_revision": args.model_revision,
        "resolved_model_revision": resolved_model_revision,
        "resolved_tokenizer_revision": resolved_model_revision,
        "dataset_name": args.dataset_name,
        "dataset_revision": args.dataset_revision,
        "resolved_dataset_revision": resolved_dataset_revision,
        "split": args.split,
        "dataset_format": DATASET_FORMAT,
        "dataset_fields": {
            "question": QUESTION_FIELD,
            "reasoning": REASONING_FIELD,
            "answer": ANSWER_FIELD,
        },
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "thinking_prefix": THINK_PREFIX,
        "answer_prefix": ANSWER_PREFIX,
        "answer_label": ANSWER_LABEL,
        "answer_label_policy": "prefix_if_missing",
        "max_samples": args.max_samples,
        "sample_seed": args.sample_seed,
        "max_length": args.max_length,
        "separator": args.separator,
        "rank_threshold": args.rank_threshold,
        "selection_threshold": args.selection_threshold,
        "lm_head_chunk_size": args.lm_head_chunk_size,
        "svd_device": args.svd_device,
        "svd_driver": args.svd_driver,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "trust_remote_code": args.trust_remote_code,
        "num_processes": num_processes,
        "svd_scope": "per_sample",
        "spectral_strength": "mean_truncated_leverage_score",
        "final_answer_policy": "exclude_from_svd_always_train",
        "eos_policy": "always_train",
        "separator_policy": "belongs_to_previous_step",
        "packing": False,
        "padding_side": "right",
    }


def _initialize_output(
    accelerator: Accelerator,
    args: argparse.Namespace,
    run_config: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        run_config = {
            **run_config,
            "local_model_sha256": fingerprint_local_model(args.model_name_or_path),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / RUN_CONFIG_FILENAME
        if config_path.exists():
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            if existing != run_config:
                raise ValueError(
                    f"Existing {config_path} does not match this run; use a new output directory"
                )
        else:
            atomic_write_json(config_path, run_config)
    accelerator.wait_for_everyone()
    run_config = json.loads(
        (output_dir / RUN_CONFIG_FILENAME).read_text(encoding="utf-8")
    )

    local_shard = shard_path(output_dir, accelerator.process_index)
    local_shard.parent.mkdir(parents=True, exist_ok=True)
    if local_shard.exists() and local_shard.stat().st_size and not args.resume:
        raise FileExistsError(f"{local_shard} already exists; pass --resume or use a new directory")
    local_shard.touch(exist_ok=True)
    return output_dir, run_config


def _load_tokenizer(args: argparse.Namespace, revision: str | None) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        revision=revision or args.model_revision,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(
    args: argparse.Namespace,
    device: torch.device,
    revision: str | None,
) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        revision=revision or args.model_revision,
        torch_dtype=_dtype_from_name(args.dtype),
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _prepare_source_indices(
    accelerator: Accelerator,
    args: argparse.Namespace,
    output_dir: Path,
    dataset: Any,
    tokenizer: Any,
) -> tuple[list[int], dict[str, Any]]:
    indices_path = output_dir / SOURCE_INDICES_FILENAME
    validation_path = output_dir / PREVALIDATION_FILENAME
    if accelerator.is_main_process and not indices_path.exists():
        candidates = list(range(len(dataset)))
        desired = None if args.max_samples <= 0 else min(args.max_samples, len(dataset))
        if desired is not None:
            random.Random(args.sample_seed).shuffle(candidates)

        valid_indices: list[int] = []
        errors: list[dict[str, Any]] = []
        for source_index in candidates:
            try:
                prepare_sample(
                    dataset[source_index],
                    tokenizer,
                    max_length=args.max_length,
                    separator=args.separator,
                )
            except SampleFormatError as error:
                errors.append({"source_index": source_index, "error": str(error)})
                continue

            valid_indices.append(source_index)
            if desired is not None and len(valid_indices) >= desired:
                break
            if len(valid_indices) % 1_000 == 0:
                LOGGER.info("Prevalidated %d usable samples", len(valid_indices))

        if desired is not None and len(valid_indices) != desired:
            raise RuntimeError(
                f"Only {len(valid_indices)} valid samples found; requested {desired}"
            )
        atomic_write_json(
            indices_path,
            {
                "source_indices": valid_indices,
                "sha256": _source_indices_hash(valid_indices),
            },
        )
        atomic_write_json(
            validation_path,
            {
                "checked_candidates": len(valid_indices) + len(errors),
                "valid_samples": len(valid_indices),
                "invalid_samples": len(errors),
                "errors": errors,
            },
        )
    accelerator.wait_for_everyone()

    source_payload = json.loads(indices_path.read_text(encoding="utf-8"))
    source_indices = [int(index) for index in source_payload["source_indices"]]
    if source_payload["sha256"] != _source_indices_hash(source_indices):
        raise ValueError(f"Corrupted source index artifact: {indices_path}")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    return source_indices, validation


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    accelerator = Accelerator()
    set_seed(args.sample_seed)
    resolved_model_revision, resolved_dataset_revision = _resolve_run_revisions(args)
    run_config = _requested_config(
        args,
        accelerator.num_processes,
        resolved_model_revision=resolved_model_revision,
        resolved_dataset_revision=resolved_dataset_revision,
    )
    output_dir, run_config = _initialize_output(accelerator, args, run_config)

    dataset = load_dataset(
        args.dataset_name,
        split=args.split,
        revision=run_config.get("resolved_dataset_revision") or args.dataset_revision,
    )
    tokenizer = _load_tokenizer(
        args,
        run_config.get("resolved_tokenizer_revision"),
    )
    source_indices, prevalidation = _prepare_source_indices(
        accelerator,
        args,
        output_dir,
        dataset,
        tokenizer,
    )
    model = _load_model(
        args,
        accelerator.device,
        run_config.get("resolved_model_revision"),
    )
    svd_device = _resolve_svd_device(args.svd_device, accelerator.device)
    local_shard = shard_path(output_dir, accelerator.process_index)
    already_done = completed_source_indices(local_shard) if args.resume else set()
    error_path = local_shard.with_name(local_shard.stem + "-errors.jsonl")

    local_processed = 0
    for sample_position, source_index in enumerate(source_indices):
        if sample_position % accelerator.num_processes != accelerator.process_index:
            continue
        if source_index in already_done:
            continue

        try:
            prepared = prepare_sample(
                dataset[source_index],
                tokenizer,
                max_length=args.max_length,
                separator=args.separator,
            )
            gradients, step_ids = capture_reasoning_gradient_matrix(
                model,
                prepared,
                device=accelerator.device,
                lm_head_chunk_size=args.lm_head_chunk_size,
            )
            if svd_device.type == "cuda":
                torch.cuda.empty_cache()
            selection = exact_spectral_selection(
                gradients.to(svd_device),
                step_ids.to(svd_device),
                num_steps=len(prepared.step_token_positions),
                rank_threshold=args.rank_threshold,
                selection_threshold=args.selection_threshold,
                svd_driver=args.svd_driver or None,
            )
            active_ranges = prepared.active_ranges(selection.selected_step_ids)
            record = MaskRecord(
                sample_position=sample_position,
                source_index=source_index,
                input_ids_hash=prepared.input_ids_hash,
                sequence_length=len(prepared.input_ids),
                truncated=prepared.truncated,
                spectral_rank=selection.rank,
                rank_cumulative_energy=selection.cumulative_energy,
                step_strengths=selection.step_strengths,
                selected_step_ids=selection.selected_step_ids,
                active_token_ranges=tuple(
                    (int(start), int(end)) for start, end in active_ranges
                ),
                reasoning_tokens=selection.reasoning_tokens,
                selected_reasoning_tokens=selection.selected_reasoning_tokens,
                final_answer_tokens=len(prepared.final_answer_positions),
                eos_tokens=len(prepared.eos_positions),
            )
            append_jsonl(local_shard, record.to_dict())
            local_processed += 1
            if args.log_every > 0 and local_processed % args.log_every == 0:
                LOGGER.info(
                    "rank=%d processed=%d source_index=%d spectral_rank=%d retention=%.4f",
                    accelerator.process_index,
                    local_processed,
                    source_index,
                    selection.rank,
                    selection.token_retention_ratio,
                )
            del gradients, step_ids, selection
            if accelerator.device.type == "cuda":
                torch.cuda.empty_cache()
        except SampleFormatError as error:
            append_jsonl(
                error_path,
                {
                    "sample_position": sample_position,
                    "source_index": source_index,
                    "error": str(error),
                },
            )
            raise RuntimeError(
                "A prevalidated sample changed format during spectral analysis"
            ) from error

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        records = merge_mask_shards(
            output_dir,
            num_processes=accelerator.num_processes,
        )
        resolved_model_revision = getattr(model.config, "_commit_hash", None)
        resolved_tokenizer_revision = (
            getattr(tokenizer, "_commit_hash", None)
            or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        )
        expected_model_revision = run_config.get("resolved_model_revision")
        expected_tokenizer_revision = run_config.get("resolved_tokenizer_revision")
        if (
            expected_model_revision
            and resolved_model_revision
            and expected_model_revision != resolved_model_revision
        ):
            raise RuntimeError("Loaded model commit differs from locked run configuration")
        if (
            expected_tokenizer_revision
            and resolved_tokenizer_revision
            and expected_tokenizer_revision != resolved_tokenizer_revision
        ):
            raise RuntimeError("Loaded tokenizer commit differs from locked run configuration")
        manifest = {
            **run_config,
            "created_at": utc_now_iso(),
            "resolved_model_revision": resolved_model_revision,
            "resolved_tokenizer_revision": resolved_tokenizer_revision,
            "source_dataset_rows": len(dataset),
            "requested_source_indices": len(source_indices),
            "source_indices_sha256": _source_indices_hash(source_indices),
            "prevalidation": prevalidation,
            "mask_file": "masks.jsonl",
            "summary": summarize_records(records),
            "software": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": _package_version("transformers"),
                "datasets": _package_version("datasets"),
                "accelerate": _package_version("accelerate"),
            },
        }
        atomic_write_json(output_dir / MANIFEST_FILENAME, manifest)
        LOGGER.info("Wrote %d masks to %s", len(records), output_dir)
    accelerator.wait_for_everyone()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
