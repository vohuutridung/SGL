# Spectral-guided Learning

Implementation of *Uncovering the Gradient Geometry of Long
CoT: A Spectral-guided Approach to Reasoning Distillation*.

This repository implements the primary method only:

1. Format each training sample with the model's native chat template.
2. Capture token loss gradients with respect to the final hidden state immediately before the LM head.
3. Run an exact, reduced, per-sample SVD in FP32.
4. Select the smallest rank whose squared singular values contain 95% of the gradient energy.
5. Compute equation (7), the mean truncated leverage score for each reasoning step.
6. Select the smallest stable, score-sorted step prefix containing 80% of the
   total step strength.
7. Full SFT on the complete sequence while masking the local losses of
   unselected reasoning steps.

## Installation

Python 3.10 or newer is required. `transformers==5.4.0` and `accelerate==1 13.0` are pinned.

```bash
python -m pip install -e ".[dev]"
```

For multi-GPU ZeRO-3 full fine-tuning:

```bash
python -m pip install -e ".[train,dev]"
```

## 1. Build spectral masks

Each process owns complete samples; no sample or SVD is split across ranks. The model is replicated once per GPU. For example, on eight GPUs:

```bash
accelerate launch --num_processes 8 -m sgl.build_masks \
  --model-name-or-path Qwen/Qwen3-4B-Base \
  --output-dir artifacts/qwen3-4b \
  --max-samples 10000 \
  --rank-threshold 0.95 \
  --selection-threshold 0.8 \
  --svd-device cuda \
  --svd-driver gesvd \
  --dtype bfloat16 \
  --attn-implementation sdpa
```

On a single GPU, run `python -m` without Accelerate. Use `--dtype float16`
if the GPU does not support BF16:

```bash
python -m sgl.build_masks \
  --model-name-or-path Qwen/Qwen3-4B-Base \
  --output-dir artifacts/qwen3-4b \
  --max-samples 10000 \
  --rank-threshold 0.95 \
  --selection-threshold 0.8 \
  --svd-device cuda \
  --svd-driver gesvd \
  --dtype bfloat16 \
  --attn-implementation sdpa
```


Exact SVD for every long response is intentionally expensive. Use
`--lm-head-chunk-size` to control logit memory. `--resume` safely resumes
completed per-rank JSONL shards.

The output directory contains:

- `run_config.json`: immutable analysis configuration.
- `shards/`: resumable rank-local results.
- `masks.jsonl`: merged masks sorted by sampled position.
- `manifest.json`: revisions, policies, software versions, and retention
  summary.

## 2. Full fine-tuning

The example below gives a global batch size of 32 automatically. With eight
processes and per-device batch size one, gradient accumulation is four.
`--deepspeed configs/deepspeed_zero3.json` is for multi-GPU full SFT.

```bash
torchrun --nproc_per_node 8 -m sgl.training \
  --mask-dir artifacts/qwen3-4b \
  --output-dir checkpoints/qwen3-4b-sgl \
  --global-batch-size 32 \
  --per-device-train-batch-size 1 \
  --num-train-epochs 6 \
  --learning-rate 5e-5 \
  --min-learning-rate 1e-5 \
  --warmup-ratio 0.1 \
  --deepspeed configs/deepspeed_zero3.json
```

On a single GPU, run this. You can reduce per-device-train-batch-size if OOM:

```bash
python -m sgl.training \
  --mask-dir artifacts/qwen3-4b \
  --output-dir checkpoints/qwen3-4b-sgl \
  --global-batch-size 32 \
  --per-device-train-batch-size 32 \
  --num-train-epochs 6 \
  --learning-rate 5e-5 \
  --min-learning-rate 1e-5 \
  --warmup-ratio 0.1
```

## 3. Evaluation

Supported benchmarks: GSM8K (`openai/gsm8k`), MATH-500 (`HuggingFaceH4/MATH-500`), AIME 2024 (`HuggingFaceH4/aime_2024`), AIME 2025 (`yentinglin/aime_2025`).

On eight GPUs:

```bash
accelerate launch --num_processes 8 -m sgl.evaluation \
  --model-name-or-path checkpoints/qwen3-4b-sgl \
  --output-dir outputs/qwen3-4b-sgl \
  --benchmarks gsm8k math500 aime24 aime25 \
  --num-generations 4 \
  --temperature 0.6 \
  --top-p 0.95 \
  --max-new-tokens 32768
```

On a single GPU:

```bash
python -m sgl.evaluation \
  --model-name-or-path checkpoints/qwen3-4b-sgl \
  --output-dir outputs/qwen3-4b-sgl \
  --benchmarks gsm8k math500 aime24 aime25 \
  --num-generations 4 \
  --temperature 0.6 \
  --top-p 0.95 \
  --max-new-tokens 32768
```

Scoring uses HuggingFace `math-verify` 0.9 conventions:

- expression extraction for GSM8K and AIME gold answers;
- LaTeX extraction for MATH-500 gold answers;
- boxed-LaTeX extraction followed by expression fallback for predictions.

Reported accuracy is mean pass@1 over K independent generations:

```text
sum(correct generation indicators) / (problems * K)
```