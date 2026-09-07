from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from datasets import load_dataset
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from sgl.artifacts import (
    append_jsonl,
    atomic_write_json,
    fingerprint_local_model,
    iter_jsonl,
)
from sgl.data import (
    DEFAULT_SYSTEM_PROMPT,
    native_assistant_end_token_id,
)

LOGGER = logging.getLogger("sgl.evaluation")
RUN_CONFIG_FILENAME = "evaluation_config.json"


@dataclass(frozen=True)
class BenchmarkSpec:
    dataset_name: str
    revision: str
    config_name: str
    split: str
    question_field: str
    gold: Callable[[dict[str, Any]], str]
    gold_is_latex: bool


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "gsm8k": BenchmarkSpec(
        dataset_name="openai/gsm8k",
        revision="740312add88f781978c0658806c59bc2815b9866",
        config_name="main",
        split="test",
        question_field="question",
        gold=lambda row: str(row["answer"]).split("####")[-1].strip(),
        gold_is_latex=False,
    ),
    "math500": BenchmarkSpec(
        dataset_name="HuggingFaceH4/MATH-500",
        revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        config_name="default",
        split="test",
        question_field="problem",
        gold=lambda row: str(row["answer"]),
        gold_is_latex=True,
    ),
    "aime24": BenchmarkSpec(
        dataset_name="HuggingFaceH4/aime_2024",
        revision="2fe88a2f1091d5048c0f36abc874fb997b3dd99a",
        config_name="default",
        split="train",
        question_field="problem",
        gold=lambda row: str(row["answer"]),
        gold_is_latex=False,
    ),
    "aime25": BenchmarkSpec(
        dataset_name="yentinglin/aime_2025",
        revision="6f71d77b0b89b9dabe07ab466c51df33f514df7f",
        config_name="default",
        split="train",
        question_field="problem",
        gold=lambda row: str(row["answer"]),
        gold_is_latex=False,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SGL checkpoints with mean pass@1 accuracy over four generations."
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=sorted(BENCHMARKS),
        default=sorted(BENCHMARKS),
    )
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=32_768)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--system-prompt")
    parser.add_argument("--system-prompt-file")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _load_system_prompt(args: argparse.Namespace) -> str:
    if args.system_prompt is not None and args.system_prompt_file is not None:
        raise ValueError("Use only one of --system-prompt and --system-prompt-file")
    if args.system_prompt is not None:
        return args.system_prompt
    if args.system_prompt_file is not None:
        return Path(args.system_prompt_file).read_text(encoding="utf-8").strip()

    return DEFAULT_SYSTEM_PROMPT


def _initialize_output(
    args: argparse.Namespace,
    accelerator: Accelerator,
    output_dir: Path,
    system_prompt: str,
    tokenizer: Any,
    model: Any,
    eos_token_ids: list[int],
    max_context_length: int | None,
) -> None:
    if accelerator.is_main_process:
        resolved_model_revision = getattr(model.config, "_commit_hash", None)
        resolved_tokenizer_revision = (
            getattr(tokenizer, "_commit_hash", None)
            or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        )
        config = {
            "schema_version": 1,
            "model_name_or_path": args.model_name_or_path,
            "model_revision": args.model_revision,
            "resolved_model_revision": resolved_model_revision,
            "resolved_tokenizer_revision": resolved_tokenizer_revision,
            "local_model_sha256": fingerprint_local_model(args.model_name_or_path),
            "benchmarks": args.benchmarks,
            "benchmark_revisions": {
                name: BENCHMARKS[name].revision for name in args.benchmarks
            },
            "num_generations": args.num_generations,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "limit": args.limit,
            "seed": args.seed,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "trust_remote_code": args.trust_remote_code,
            "num_processes": accelerator.num_processes,
            "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
            "eos_token_ids": eos_token_ids,
            "max_context_length": max_context_length,
            "software": {
                "torch": torch.__version__,
                "transformers": importlib.metadata.version("transformers"),
                "datasets": importlib.metadata.version("datasets"),
                "accelerate": importlib.metadata.version("accelerate"),
                "math_verify": importlib.metadata.version("math-verify"),
            },
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / RUN_CONFIG_FILENAME
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != config:
                raise ValueError(
                    f"Existing {path} is incompatible; use a new output directory"
                )
        else:
            atomic_write_json(path, config)
    accelerator.wait_for_everyone()


@lru_cache(maxsize=2)
def _make_metric(gold_is_latex: bool) -> Callable[..., Any]:
    gold_target = (
        (LatexExtractionConfig(boxed_match_priority=0),)
        if gold_is_latex
        else (ExprExtractionConfig(),)
    )
    return math_metric(
        gold_extraction_target=gold_target,
        pred_extraction_target=(
            LatexExtractionConfig(),
            ExprExtractionConfig(),
        ),
        aggregation_function=max,
    )


def grade_response(
    response: str,
    gold: str,
    *,
    gold_is_latex: bool,
) -> tuple[bool, Any]:
    metric = _make_metric(gold_is_latex)
    metric_gold = f"${gold}$" if gold_is_latex else gold
    score, extracted = metric([metric_gold], [response])
    return bool(score == 1), extracted


def generation_eos_token_ids(model: Any, tokenizer: Any) -> list[int]:
    candidates: list[int] = []
    for value in (
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
        getattr(getattr(model, "config", None), "eos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        native_assistant_end_token_id(tokenizer),
    ):
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            candidates.extend(int(token_id) for token_id in value)
        else:
            candidates.append(int(value))
    return list(dict.fromkeys(candidates))


def _load_completed_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {int(payload["index"]) for payload in iter_jsonl(path)}


def _generate_responses(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    device: torch.device,
    num_generations: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
    eos_token_ids: list[int],
    max_context_length: int | None,
) -> list[str]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(encoded, Mapping):
        input_ids = encoded["input_ids"]
    else:
        input_ids = encoded
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    effective_max_new_tokens = max_new_tokens
    if max_context_length is not None:
        effective_max_new_tokens = min(
            max_new_tokens,
            max_context_length - input_ids.shape[1],
        )
    if effective_max_new_tokens <= 0:
        raise ValueError(
            f"Prompt length {input_ids.shape[1]} leaves no generation room under "
            f"max context {max_context_length}"
        )

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_generations,
            max_new_tokens=effective_max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_ids,
            use_cache=True,
        )
    prompt_length = input_ids.shape[1]
    return tokenizer.batch_decode(
        generated[:, prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _merge_benchmark_shards(
    output_dir: Path,
    benchmark: str,
    num_processes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for process_index in range(num_processes):
        shard = output_dir / "shards" / f"{benchmark}-rank-{process_index:05d}.jsonl"
        if not shard.exists():
            raise FileNotFoundError(f"Missing evaluation shard: {shard}")
        rows.extend(iter_jsonl(shard))
    rows.sort(key=lambda row: int(row["index"]))

    indices = [int(row["index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate {benchmark} evaluation indices")

    destination = output_dir / f"{benchmark}.jsonl"
    temporary = destination.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, destination)
    return rows


def _evaluate_benchmark(
    args: argparse.Namespace,
    accelerator: Accelerator,
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    benchmark: str,
    output_dir: Path,
    eos_token_ids: list[int],
    max_context_length: int | None,
) -> dict[str, Any] | None:
    spec = BENCHMARKS[benchmark]
    dataset = load_dataset(
        spec.dataset_name,
        spec.config_name,
        split=spec.split,
        revision=spec.revision,
    )
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    shard = (
        output_dir
        / "shards"
        / f"{benchmark}-rank-{accelerator.process_index:05d}.jsonl"
    )
    shard.parent.mkdir(parents=True, exist_ok=True)
    if shard.exists() and shard.stat().st_size and not args.resume:
        raise FileExistsError(f"{shard} exists; pass --resume or use a new output directory")
    shard.touch(exist_ok=True)
    completed = _load_completed_indices(shard) if args.resume else set()

    local_count = 0
    for index, row in enumerate(dataset):
        if index % accelerator.num_processes != accelerator.process_index:
            continue
        if index in completed:
            continue

        question = str(row[spec.question_field])
        gold = spec.gold(row)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        responses = _generate_responses(
            model,
            tokenizer,
            messages,
            device=accelerator.device,
            num_generations=args.num_generations,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed + index,
            eos_token_ids=eos_token_ids,
            max_context_length=max_context_length,
        )

        correct: list[bool] = []
        extracted: list[Any] = []
        errors: list[str | None] = []
        for response in responses:
            try:
                is_correct, parsed = grade_response(
                    response,
                    gold,
                    gold_is_latex=spec.gold_is_latex,
                )
                correct.append(is_correct)
                extracted.append(str(parsed))
                errors.append(None)
            except Exception as error:  # A malformed generation must score as wrong.
                correct.append(False)
                extracted.append(None)
                errors.append(repr(error))

        append_jsonl(
            shard,
            {
                "benchmark": benchmark,
                "index": index,
                "question": question,
                "gold": gold,
                "responses": responses,
                "correct": correct,
                "extracted": extracted,
                "errors": errors,
                "pass_at_1_over_generations": sum(correct) / len(correct),
            },
        )
        local_count += 1
        if args.log_every > 0 and local_count % args.log_every == 0:
            LOGGER.info(
                "rank=%d benchmark=%s processed=%d",
                accelerator.process_index,
                benchmark,
                local_count,
            )

    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return None

    rows = _merge_benchmark_shards(output_dir, benchmark, accelerator.num_processes)
    total_generations = sum(len(row["correct"]) for row in rows)
    correct_generations = sum(sum(row["correct"]) for row in rows)
    return {
        "benchmark": benchmark,
        "problems": len(rows),
        "generations_per_problem": args.num_generations,
        "correct_generations": correct_generations,
        "total_generations": total_generations,
        "pass_at_1_accuracy": (
            correct_generations / total_generations if total_generations else 0.0
        ),
        "dataset_name": spec.dataset_name,
        "dataset_revision": spec.revision,
        "dataset_config": spec.config_name,
        "dataset_split": spec.split,
    }


def run(args: argparse.Namespace) -> None:
    if args.num_generations <= 0:
        raise ValueError("--num-generations must be positive")
    if args.temperature <= 0:
        raise ValueError("The requested sampled evaluation needs temperature > 0")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    accelerator = Accelerator()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    system_prompt = _load_system_prompt(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        revision=args.model_revision,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        revision=args.model_revision,
        torch_dtype=_dtype_from_name(args.dtype),
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.to(accelerator.device)
    model.eval()
    eos_token_ids = generation_eos_token_ids(model, tokenizer)
    max_context_length = getattr(model.config, "max_position_embeddings", None)
    _initialize_output(
        args,
        accelerator,
        output_dir,
        system_prompt,
        tokenizer,
        model,
        eos_token_ids,
        max_context_length,
    )

    summaries: list[dict[str, Any]] = []
    for benchmark in args.benchmarks:
        summary = _evaluate_benchmark(
            args,
            accelerator,
            model,
            tokenizer,
            system_prompt,
            benchmark,
            output_dir,
            eos_token_ids,
            max_context_length,
        )
        if summary is not None:
            summaries.append(summary)
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        atomic_write_json(
            output_dir / "summary.json",
            {
                "model_name_or_path": args.model_name_or_path,
                "model_revision": args.model_revision,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "num_generations": args.num_generations,
                "eos_token_ids": eos_token_ids,
                "max_context_length": max_context_length,
                "metric": "mean pass@1 accuracy over independent generations",
                "benchmarks": summaries,
            },
        )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
