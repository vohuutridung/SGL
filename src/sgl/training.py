from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from accelerate.utils import DistributedType
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from sgl.artifacts import (
    MANIFEST_FILENAME,
    MASKS_FILENAME,
    MaskRecord,
    fingerprint_local_model,
    load_mask_records,
)
from sgl.data import (
    IGNORE_INDEX,
    SpectralDataCollator,
    native_assistant_end_token_id,
    prepare_sample,
)


class SpectralMaskDataset(Dataset):
    def __init__(
        self,
        source_dataset: Any,
        records: list[MaskRecord],
        tokenizer: Any,
        manifest: dict[str, Any],
    ) -> None:
        self.source_dataset = source_dataset
        self.records = records
        self.tokenizer = tokenizer
        self.manifest = manifest
        for record in records:
            if not 0 <= record.source_index < len(source_dataset):
                raise IndexError(f"source_index={record.source_index} is outside the dataset")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        record = self.records[index]
        prepared = prepare_sample(
            self.source_dataset[record.source_index],
            self.tokenizer,
            max_length=int(self.manifest["max_length"]),
            separator=str(self.manifest["separator"]),
            final_marker=str(self.manifest["final_marker"]),
            require_final_answer=True,
        )
        if prepared.input_ids_hash != record.input_ids_hash:
            raise RuntimeError(
                "Tokenized sample hash differs from spectral analysis for "
                f"source_index={record.source_index}. Check model/tokenizer revision "
                "and chat template."
            )
        if len(prepared.input_ids) != record.sequence_length:
            raise RuntimeError(
                f"Sequence length changed for source_index={record.source_index}"
            )
        labels = prepared.build_labels(record.active_token_ranges)
        return {
            "input_ids": prepared.input_ids,
            "labels": labels,
        }


class PerSampleMaskedTrainer(Trainer):
    """Trainer implementing Eq. (9) independently for every sample."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Qwen forwards accept generic loss kwargs, so Trainer would otherwise
        # assume our custom loss consumed num_items_in_batch and skip dividing
        # by the actual gradient-accumulation window. Mark it explicitly false;
        # this is also required when DeepSpeed sets scale_wrt_gas=False.
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        del num_items_in_batch
        labels = inputs["labels"]
        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        outputs = model(**model_inputs, use_cache=False)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        active = shift_labels.ne(IGNORE_INDEX)
        active_per_sample = active.sum(dim=1)
        if torch.any(active_per_sample == 0):
            raise ValueError("Every sample must contain at least one active causal target")

        safe_labels = shift_labels.masked_fill(~active, 0)
        token_loss = F.cross_entropy(
            shift_logits.float().view(-1, shift_logits.shape[-1]),
            safe_labels.view(-1),
            reduction="none",
        ).view_as(shift_labels)
        per_sample_loss = (token_loss * active).sum(dim=1) / active_per_sample
        loss = per_sample_loss.mean()
        # With model_accepts_loss_kwargs=False, Trainer first divides by the
        # actual accumulation-window size. Accelerate then divides once more
        # by configured GAS on non-DeepSpeed backends, but not on DeepSpeed
        # (where Trainer passes scale_wrt_gas=False). Compensate only for the
        # non-DeepSpeed division so both paths optimize the same objective,
        # including a short final accumulation window.
        accelerator = getattr(self, "accelerator", None)
        if (
            accelerator is not None
            and accelerator.distributed_type != DistributedType.DEEPSPEED
        ):
            loss = loss * self.args.gradient_accumulation_steps
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full fine-tuning with fixed SGL masks.")
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--allow-model-path-override", action="store_true")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--num-train-epochs", type=float, default=6.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--deepspeed")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    return parser.parse_args()


def _dtype_flags(dtype: str) -> tuple[torch.dtype, bool, bool]:
    if dtype == "bfloat16":
        return torch.bfloat16, True, False
    if dtype == "float16":
        return torch.float16, False, True
    return torch.float32, False, False


def _gradient_accumulation_steps(args: argparse.Namespace) -> int:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    micro_global = world_size * args.per_device_train_batch_size
    if args.global_batch_size % micro_global:
        raise ValueError(
            "global_batch_size must be divisible by "
            "WORLD_SIZE * per_device_train_batch_size"
        )
    steps = args.global_batch_size // micro_global
    if steps < 1:
        raise ValueError("global_batch_size is smaller than one distributed microbatch")
    return steps


def _make_training_arguments(
    args: argparse.Namespace,
    *,
    gradient_accumulation_steps: int,
    bf16: bool,
    fp16: bool,
) -> TrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "do_train": True,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "cosine_with_min_lr",
        "lr_scheduler_kwargs": {"min_lr": args.min_learning_rate},
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_epsilon": args.adam_epsilon,
        "max_grad_norm": args.max_grad_norm,
        "optim": args.optim,
        "bf16": bf16,
        "fp16": fp16,
        "tf32": args.tf32,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "use_cache": False,
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "report_to": "none",
        "seed": args.seed,
        "data_seed": args.seed,
        "dataloader_drop_last": True,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": True,
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "ddp_find_unused_parameters": False,
        "average_tokens_across_devices": False,
    }
    if args.deepspeed:
        kwargs["deepspeed"] = args.deepspeed

    supported = inspect.signature(TrainingArguments).parameters
    return TrainingArguments(**{key: value for key, value in kwargs.items() if key in supported})


def run(args: argparse.Namespace) -> None:
    mask_dir = Path(args.mask_dir)
    manifest = json.loads((mask_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    records = load_mask_records(mask_dir / manifest.get("mask_file", MASKS_FILENAME))
    if not records:
        raise ValueError("Mask artifact contains no training samples")

    analysis_model = str(manifest["model_name_or_path"])
    model_name_or_path = args.model_name_or_path or analysis_model
    expected_local_fingerprint = manifest.get("local_model_sha256")
    if model_name_or_path != analysis_model and not args.allow_model_path_override:
        raise ValueError(
            "Training must start from the same model used for spectral analysis. "
            "Pass --allow-model-path-override only for a byte-equivalent local copy."
        )
    if (
        model_name_or_path != analysis_model
        and expected_local_fingerprint is None
    ):
        raise ValueError(
            "A Hub model artifact cannot be overridden with an unverifiable local path. "
            "Use the same Hub model ID, or build masks from that local path."
        )
    if expected_local_fingerprint is not None:
        actual_local_fingerprint = fingerprint_local_model(model_name_or_path)
        if actual_local_fingerprint != expected_local_fingerprint:
            raise ValueError(
                "Local model/tokenizer files changed since spectral analysis"
            )
    analysis_trust_remote_code = bool(manifest.get("trust_remote_code", False))
    if args.trust_remote_code != analysis_trust_remote_code:
        raise ValueError(
            "--trust-remote-code must match the spectral-analysis run"
        )

    set_seed(args.seed)
    torch_dtype, bf16, fp16 = _dtype_flags(args.dtype)
    model_revision = (
        manifest.get("resolved_model_revision")
        or manifest.get("model_revision")
        or "main"
    )
    tokenizer_revision = (
        manifest.get("resolved_tokenizer_revision")
        or manifest.get("resolved_model_revision")
        or manifest.get("model_revision")
        or "main"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        revision=tokenizer_revision,
        use_fast=True,
        trust_remote_code=analysis_trust_remote_code,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    source_dataset = load_dataset(
        manifest["dataset_name"],
        split=manifest["split"],
        revision=(
            manifest.get("resolved_dataset_revision")
            or manifest["dataset_revision"]
        ),
    )
    train_dataset = SpectralMaskDataset(
        source_dataset,
        records,
        tokenizer,
        manifest,
    )
    collator = SpectralDataCollator(tokenizer=tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        revision=model_revision,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=analysis_trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    configured_eos = getattr(model.generation_config, "eos_token_id", None)
    eos_token_ids = (
        list(configured_eos)
        if isinstance(configured_eos, (list, tuple))
        else [configured_eos]
        if configured_eos is not None
        else []
    )
    eos_token_ids.append(native_assistant_end_token_id(tokenizer))
    model.generation_config.eos_token_id = list(dict.fromkeys(eos_token_ids))

    gradient_accumulation_steps = _gradient_accumulation_steps(args)
    training_args = _make_training_arguments(
        args,
        gradient_accumulation_steps=gradient_accumulation_steps,
        bf16=bf16,
        fp16=fp16,
    )
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": collator,
    }
    trainer_signature = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = PerSampleMaskedTrainer(**trainer_kwargs)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
