# Spectral-guided Learning

Implementation of *Uncovering the Gradient Geometry of Long
CoT: A Spectral-guided Approach to Reasoning Distillation*.

This repository implements the primary method only:

1. Format each training sample with the model's native chat template.
2. Capture token loss gradients with respect to the final hidden state
   immediately before the LM head.
3. Run an exact, reduced, per-sample SVD in FP32.
4. Select the smallest rank whose squared singular values contain 95% of the
   gradient energy.
5. Compute equation (7), the mean truncated leverage score for each reasoning
   step.
6. Select the smallest stable, score-sorted step prefix containing 80% of the
   total step strength.
7. Full-fine-tune on the complete sequence while masking the local losses of
   unselected reasoning steps.

## Fixed interpretation

The following choices are explicit in this implementation:

- Dataset: `Elliott/Openr1-Math-46k-8192`.
- Relevant fields: `prompt` (system and user messages) and `target` (one
  assistant message).
- The reasoning block ends at `</think>`. Text after `</think>\n\n` is the
  final solution.
- Reasoning steps are split at `\n\n`.
- Final-solution and native assistant end-of-turn/EOS tokens are excluded from
  SVD and always supervised. For Qwen this is `<|im_end|>`, which is distinct
  from tokenizer-level `<|endoftext|>`.
- Prompt, assistant-prefix, and non-EOS template tokens have label `-100`.
- Native chat template, no packing, right padding.
- Maximum length 32,768. Overlength samples keep the longest prefix of complete reasoning steps that fits together with the complete final answer and native end-of-turn token. A sample is rejected only if even that mandatory structure cannot fit.
- Spectral masks are computed once at the initial student weights and are
  immutable during SFT.
- Loss is normalized independently by each sample's selected token count,
  then averaged across samples.

## Installation

Python 3.10 or newer is required.

`transformers==5.4.0` and `accelerate==1.13.0` are pinned because the custom
per-sample objective relies on their actual-window gradient-accumulation
semantics, including a short final window.

```bash
python -m pip install -e ".[dev]"
```

For multi-GPU ZeRO-3 full fine-tuning:

```bash
python -m pip install -e ".[train,dev]"
```

## 1. Build spectral masks

Each process owns complete samples; no sample or SVD is split across ranks.
The model is replicated once per GPU. For example:

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

Important implementation details:
- The decoder forward pass runs without parameter gradients.
- LM-head logits are processed in chunks to avoid materializing
  `[sequence, vocabulary]` for the entire response.
- Each gradient chunk is cast to FP32 before forming `G`.
- `torch.linalg.svd(G, full_matrices=False)` computes the full reduced SVD.

Exact SVD for every long response is intentionally expensive. Use
`--lm-head-chunk-size` to control logit memory. `--resume` safely resumes
completed per-rank JSONL shards.

The output directory contains:

- `run_config.json`: immutable analysis configuration.
- `shards/`: resumable rank-local results.
- `masks.jsonl`: merged masks sorted by sampled position.
- `manifest.json`: revisions, policies, software versions, and retention
  summary.

Hub `main` references are resolved to immutable model, tokenizer, and dataset
commits before the first shard is written. Resume always reuses those commits.

## 2. Full fine-tuning

The example below gives a global batch size of 32 automatically. With eight
processes and per-device batch size one, gradient accumulation is four.

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

Defaults not reported by the paper follow standard Transformers/LLaMA-Factory
behavior:

- AdamW beta1 0.9, beta2 0.999, epsilon `1e-8`.
- Weight decay 0.
- Maximum gradient norm 1.
- BF16, gradient checkpointing, no label smoothing.
- Cosine-with-minimum-LR scheduler.

## 3. Evaluation

Supported benchmarks:

- GSM8K: `openai/gsm8k`, `main`, test split.
- MATH-500: `HuggingFaceH4/MATH-500`, test split.
- AIME 2024: `HuggingFaceH4/aime_2024`, train split containing all 30 tasks.
- AIME 2025: `yentinglin/aime_2025`, train split containing all 30 tasks.

All four benchmark repositories are pinned to the revisions recorded in
`src/sgl/evaluation.py`.

The evaluator reuses the training dataset's system prompt and the checkpoint's
native chat template. It samples four responses per problem with temperature
0.6, top-p 0.95, and up to 32,768 new tokens:

```bash
accelerate launch --num_processes 8 -m sgl.evaluation \
  --model-name-or-path checkpoints/qwen3-4b-sgl \
  --output-dir outputs/qwen3-4b-sgl \
  --benchmarks gsm8k math500 aime24 aime25 \
  --num-generations 32 \
  --temperature 0.6 \
  --top-p 0.95 \
  --max-new-tokens 32768
```

Generation stops on either the model EOS token or the native assistant
end-of-turn token (for Qwen, `<|im_end|>`), and the requested new-token budget
is capped so prompt plus completion cannot exceed the model context window.

Scoring uses Hugging Face `math-verify` 0.9 conventions:

- expression extraction for GSM8K and AIME gold answers;
- LaTeX extraction for MATH-500 gold answers;
- boxed-LaTeX extraction followed by expression fallback for predictions.

Reported accuracy is mean pass@1 over the four independent generations:

```text
sum(correct generation indicators) / (problems * 4)
```

It is not pass@4 and does not use majority voting.

## Tests

```bash
pytest
```

Tests cover:

- `\n\n` ownership and final-answer boundaries;
- chat-template token alignment, right padding, and mandatory EOS/final masks;
- causal hidden/target shift;
- exact rank selection and stable tie order;
- leverage score versus projection-energy behavior;
- per-sample rather than flattened-token loss normalization;
- direct zero gradient at masked local-loss positions;
- distributed mask-shard merging;
- numeric and LaTeX benchmark grading.
