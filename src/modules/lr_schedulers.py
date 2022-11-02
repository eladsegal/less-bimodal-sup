from torch.optim import Optimizer
import pytorch_lightning as pl
from transformers import get_scheduler


def get_hf_scheduler(name: str, optimizer: Optimizer, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
    return get_scheduler(
        name=name,
        optimizer=optimizer,
        num_warmup_steps=pl_module._num_warmup_steps,
        num_training_steps=trainer.estimated_stepping_batches,
    )
