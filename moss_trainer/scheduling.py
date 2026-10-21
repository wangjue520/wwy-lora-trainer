"""Epoch boundaries shared by training, inspection and checkpoint validation."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TrainingPlan:
    batches_per_epoch: int
    updates_per_epoch: int
    max_steps: int
    max_epochs: int
    save_every_epochs: int
    save_every_steps: int
    warmup_steps: int

    def should_save(self, step, next_epoch, next_batch):
        if step == self.max_steps:
            return True
        if self.save_every_epochs:
            return next_batch == 0 and next_epoch % self.save_every_epochs == 0
        return step % self.save_every_steps == 0


def training_plan(training, batch_count):
    if type(batch_count) is not int or batch_count < 1:
        raise ValueError("Training requires at least one bucket batch per epoch")
    updates = (batch_count + training.gradient_accumulation_steps - 1) // training.gradient_accumulation_steps
    max_steps = training.max_epochs * updates if training.max_epochs else training.max_steps
    max_epochs = training.max_epochs or (max_steps + updates - 1) // updates
    warmup = training.warmup_steps if training.warmup_steps >= 0 else math.ceil(updates * training.warmup_ratio)
    if warmup > max_steps:
        raise ValueError("Warmup exceeds the total number of optimizer updates")
    if training.scheduler == "cosine_with_restarts":
        if max_steps - warmup < training.cosine_restarts + 1:
            raise ValueError("Each cosine cycle requires at least one post-warmup optimizer update")
    return TrainingPlan(batch_count, updates, max_steps, max_epochs,
                        training.save_every_epochs, training.save_every, warmup)


def checkpoint_name(config, step, next_epoch, next_batch):
    if config.training.max_epochs:
        if next_batch != 0 or next_epoch < 1:
            raise ValueError("Epoch checkpoints can only be saved after a complete epoch")
        return f"epoch-{next_epoch:06d}"
    return f"step-{step:08d}"


def build_optimizer(parameters, training):
    """Use native PyTorch optimizers; keep AdamW's existing defaults."""
    import torch
    classes = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam,
               "sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop}
    options = {"lr": training.learning_rate, "weight_decay": training.weight_decay,
               "foreach": False}
    if training.optimizer == "sgd":
        options["momentum"] = 0.9
    return classes[training.optimizer](parameters, **options)


def build_scheduler(optimizer, training, plan):
    """Index is completed optimizer updates; warmup runs only once."""
    import torch
    warmup = plan.warmup_steps

    def multiplier(step):
        if step < warmup:
            return (step + 1) / warmup
        if training.scheduler == "constant":
            return 1.0
        progress = min(1.0, max(0.0, (step - warmup) / max(1, plan.max_steps - warmup)))
        if training.scheduler == "linear":
            return 1.0 - progress
        if progress >= 1.0:
            return 0.0
        if training.scheduler == "cosine_with_restarts":
            progress = (progress * (training.cosine_restarts + 1)) % 1.0
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
