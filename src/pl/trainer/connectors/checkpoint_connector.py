from typing import Any, Dict, Optional

import torch
from torchmetrics import Metric

from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector
from pytorch_lightning.utilities.types import _PATH


class MyCheckpointConnector(CheckpointConnector):
    # Copied from checkpoint_connector, except dropping the fault_tolerant_training condition
    def _get_lightning_module_state_dict(self) -> Dict[str, torch.Tensor]:
        metrics = [m for m in self.trainer.lightning_module.modules() if isinstance(m, Metric)]

        for metric in metrics:
            metric.persistent(True)
            metric.sync()

        state_dict = self.trainer.strategy.lightning_module_state_dict()

        for metric in metrics:
            # sync can be a no-op (e.g. on cpu) so `unsync` would raise a user error exception if we don't check
            if metric._is_synced:
                metric.unsync()

        return state_dict

    # TODO: Should be removed once fault-tolerant training is implemented properly and mid-epoch checkpoints can be resumed
    def save_checkpoint(self, filepath: _PATH, weights_only: bool = False) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            filepath: write-target file's path
            weights_only: saving model weights only
        """
        if weights_only or self.trainer.fit_loop.epoch_loop.done:
            super().save_checkpoint(filepath, weights_only)
