from typing import Any, Optional

import time

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT


class UploadOutput(Callback):
    def __init__(self, interval: int) -> None:
        super().__init__()
        self._interval = interval
        self._last_upload_time = time.time()

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self._save_if_needed(trainer)

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self._save_if_needed(trainer)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_epoch_end(trainer, pl_module)
        self._save_if_needed(trainer)

    def _save_if_needed(self, trainer: "pl.Trainer"):
        if time.time() - self._last_upload_time > self._interval:
            trainer.upload_files(["out.log"])
            self._last_upload_time = time.time()
