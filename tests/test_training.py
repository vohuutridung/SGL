from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from accelerate.utils import DistributedType
from torch import nn
from transformers import Trainer

from sgl.artifacts import SCHEMA_VERSION
from sgl.data import (
    ANSWER_FIELD,
    ANSWER_LABEL,
    ANSWER_PREFIX,
    DATASET_FORMAT,
    DEFAULT_SYSTEM_PROMPT,
    IGNORE_INDEX,
    QUESTION_FIELD,
    REASONING_FIELD,
    THINK_PREFIX,
)
from sgl.training import PerSampleMaskedTrainer, validate_manifest_format


class PositionLogitModel(nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits = nn.Parameter(logits)

    def forward(self, input_ids, attention_mask, use_cache):
        del input_ids, attention_mask, use_cache
        return SimpleNamespace(logits=self.logits)


def valid_manifest_format():
    return {
        "schema_version": SCHEMA_VERSION,
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
    }


def test_manifest_format_accepts_current_s1k_schema():
    validate_manifest_format(valid_manifest_format())


def test_manifest_format_rejects_old_schema_and_changed_format():
    old = valid_manifest_format()
    old["schema_version"] = 1
    with pytest.raises(ValueError, match="rebuild the spectral masks"):
        validate_manifest_format(old)

    changed = valid_manifest_format()
    changed["dataset_fields"] = {"question": "prompt"}
    with pytest.raises(ValueError, match="dataset_fields"):
        validate_manifest_format(changed)


def test_trainer_forces_gradient_accumulation_loss_scaling(monkeypatch):
    def fake_init(self, *args, **kwargs):
        del args, kwargs
        self.model_accepts_loss_kwargs = True

    monkeypatch.setattr(Trainer, "__init__", fake_init)
    trainer = PerSampleMaskedTrainer()
    assert trainer.model_accepts_loss_kwargs is False


def test_masked_loss_normalizes_each_sample_independently():
    logits = torch.tensor(
        [
            [
                [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]],
            ],
            [
                [[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]],
                [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
                [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
            ],
        ]
    )
    # Pick one [sequence, vocab] matrix for each sample.
    logits = torch.stack([logits[0, :, 0, :], logits[1, :, 1, :]])
    model = PositionLogitModel(logits)
    labels = torch.tensor(
        [
            [IGNORE_INDEX, 0, IGNORE_INDEX],
            [IGNORE_INDEX, 1, 2],
        ]
    )
    inputs = {
        "input_ids": torch.ones((2, 3), dtype=torch.long),
        "attention_mask": torch.ones((2, 3), dtype=torch.long),
        "labels": labels,
    }

    trainer = object.__new__(PerSampleMaskedTrainer)
    loss = trainer.compute_loss(model, inputs)

    shifted_logits = logits[:, :-1, :]
    sample_zero = F.cross_entropy(shifted_logits[0, :1].float(), torch.tensor([0]))
    sample_one = F.cross_entropy(
        shifted_logits[1].float(),
        torch.tensor([1, 2]),
        reduction="mean",
    )
    expected = (sample_zero + sample_one) / 2
    assert torch.allclose(loss, expected)

    flat_token_mean = (
        sample_zero
        + F.cross_entropy(
            shifted_logits[1].float(),
            torch.tensor([1, 2]),
            reduction="sum",
        )
    ) / 3
    assert not torch.allclose(loss, flat_token_mean)


def test_non_deepspeed_loss_compensates_accelerate_gas_scaling():
    logits = torch.zeros((1, 3, 3))
    inputs = {
        "input_ids": torch.ones((1, 3), dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "labels": torch.tensor([[IGNORE_INDEX, 1, 2]]),
    }
    baseline = object.__new__(PerSampleMaskedTrainer)
    raw_loss = baseline.compute_loss(PositionLogitModel(logits.clone()), inputs)

    non_deepspeed = object.__new__(PerSampleMaskedTrainer)
    non_deepspeed.accelerator = SimpleNamespace(
        distributed_type=DistributedType.NO
    )
    non_deepspeed.args = SimpleNamespace(gradient_accumulation_steps=4)
    scaled = non_deepspeed.compute_loss(PositionLogitModel(logits.clone()), inputs)
    assert torch.allclose(scaled, raw_loss * 4)

    deepspeed = object.__new__(PerSampleMaskedTrainer)
    deepspeed.accelerator = SimpleNamespace(
        distributed_type=DistributedType.DEEPSPEED
    )
    deepspeed.args = SimpleNamespace(gradient_accumulation_steps=4)
    unscaled = deepspeed.compute_loss(PositionLogitModel(logits.clone()), inputs)
    assert torch.allclose(unscaled, raw_loss)


def test_masked_position_has_no_direct_logit_gradient():
    logits = torch.zeros((1, 4, 3))
    model = PositionLogitModel(logits)
    inputs = {
        "input_ids": torch.ones((1, 4), dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "labels": torch.tensor([[IGNORE_INDEX, 1, IGNORE_INDEX, 2]]),
    }
    trainer = object.__new__(PerSampleMaskedTrainer)
    loss = trainer.compute_loss(model, inputs)
    loss.backward()

    # logits[:, i] predicts labels[:, i + 1]; label position 2 is masked.
    assert torch.count_nonzero(model.logits.grad[0, 1]) == 0
    assert torch.count_nonzero(model.logits.grad[0, 0]) > 0
    assert torch.count_nonzero(model.logits.grad[0, 2]) > 0
