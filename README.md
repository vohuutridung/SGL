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
7. LoRA SFT on the complete sequence while masking the local losses of
   unselected reasoning steps.

## Training data format

Training uses all 1,000 rows from `simplescaling/s1K-1.1`. Each raw row is mapped as follows:

- `question` becomes the user message.
- `deepseek_thinking_trajectory` becomes the reasoning trajectory and is split
  into steps at blank lines (`\n\n`).
- `deepseek_attempt` becomes the final answer. It is prefixed with `Answer: `
  when that label is absent.

The formatter adds the Qwen system prompt and wraps the two assistant regions
with `<|im_start|>think` and `<|im_start|>answer` before applying the model's
native chat template. These tags are not expected in the raw dataset. Only
reasoning steps participate in spectral selection; the answer delimiter,
complete final answer, and native assistant end-of-turn token are always
supervised.

## Installation

Python 3.10 or newer is required. `transformers==5.4.0` and `accelerate==1.13.0` are pinned.

```bash
python -m pip install -e ".[dev]"
```

For optional multi-GPU ZeRO-3 LoRA fine-tuning:

```bash
python -m pip install -e ".[train,dev]"
```

## 1. Build spectral masks

Each process owns complete samples; no sample or SVD is split across ranks. The model is replicated once per GPU. For example, on eight GPUs:

```bash
accelerate launch --num_processes 8 -m sgl.build_masks \
  --model-name-or-path Qwen/Qwen2.5-7B-Instruct \
  --output-dir artifacts/qwen2.5-7b \
  --max-samples 1000 \
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
  --model-name-or-path Qwen/Qwen2.5-7B-Instruct \
  --output-dir artifacts/qwen2.5-7b \
  --max-samples 1000 \
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

## 2. LoRA fine-tuning

The example below gives a global batch size of 32 automatically. With eight
processes and per-device batch size one, gradient accumulation is four.
`--deepspeed configs/deepspeed_zero3.json` is for multi-GPU training.

```bash
torchrun --nproc_per_node 8 -m sgl.training \
  --mask-dir artifacts/qwen2.5-7b \
  --output-dir checkpoints/qwen2.5-7b-sgl \
  --model-name-or-path Qwen/Qwen2.5-7B-Instruct \
  --global-batch-size 32 \
  --per-device-train-batch-size 1 \
  --num-train-epochs 6 \
  --learning-rate 5e-5 \
  --warmup-ratio 0.1 \
  --deepspeed configs/deepspeed_zero3.json
```

On a single NVIDIA B200 180 GB GPU, use a microbatch of two and four
gradient-accumulation steps, for an effective batch size of 8. If an unusually
long sample causes an out-of-memory error, reduce the microbatch to one and keep
the global batch size at 8.

```bash
python -m sgl.training \
  --mask-dir artifacts/qwen2.5-7b \
  --output-dir checkpoints/qwen2.5-7b-sgl \
  --model-name-or-path Qwen/Qwen2.5-7B-Instruct \
  --global-batch-size 8 \
  --per-device-train-batch-size 2 \
  --num-train-epochs 6 \
  --learning-rate 5e-5 \
  --warmup-ratio 0.1
```

The training recipe uses LoRA rank 16, alpha 16, dropout 0.05, and targets
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
`down_proj`. It uses AdamW with betas `(0.9, 0.999)`, epsilon `1e-8`, no
weight decay, and a cosine `LambdaLR` schedule with a 0.1 warmup ratio.
Artifacts and checkpoints are saved only to local paths.

## 3. Evaluation

Evaluation uses MATH-500 (500 problems, `HuggingFaceH4/MATH-500`),
AIME 2025 (30 problems, `yentinglin/aime_2025`), AIME 2024 (30 problems,
`HuggingFaceH4/aime_2024`), and AMC12 (83 problems,
`AI-MO/aimo-validation-amc`).

On eight GPUs:

```bash
accelerate launch --num_processes 8 -m sgl.evaluation \
  --model-name-or-path checkpoints/qwen2.5-7b-sgl \
  --output-dir outputs/qwen2.5-7b-sgl \
  --benchmarks aime25 aime24 amc12 math500 \
  --num-generations 4 \
  --temperature 0.6 \
  --top-p 0.95 \
  --max-new-tokens 32768
```

On a single GPU:

```bash
python -m sgl.evaluation \
  --model-name-or-path checkpoints/qwen2.5-7b-sgl \
  --output-dir outputs/qwen2.5-7b-sgl \
  --benchmarks aime25 aime24 amc12 math500 \
  --num-generations 4 \
  --temperature 0.6 \
  --top-p 0.95 \
  --max-new-tokens 32768
```

Scoring uses HuggingFace `math-verify` 0.9 conventions:

- expression extraction for AIME and AMC12 gold answers;
- LaTeX extraction for MATH-500 gold answers;
- boxed-LaTeX extraction followed by expression fallback for predictions.

Reported metrics:

- **pass@1**: mean accuracy over K independent generations
  `sum(correct) / (problems * K)`
- **pass@3**: unbiased estimator averaged over problems (needs `K >= 3`, default is 4)
  `1 - C(n - c, 3) / C(n, 3)` per problem, then mean
