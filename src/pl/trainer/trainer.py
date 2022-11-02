from typing import Any, List, Optional, Union
import types
import inspect
import math

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, LoggerCollection
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from src.pl.trainer.connectors.data_connector import _wrap_resolve_sampler
from src.pl.trainer.connectors.checkpoint_connector import MyCheckpointConnector
from src.pl.trainer.connectors.signal_connector import MySignalConnector

import logging

logger = logging.getLogger(__name__)


class MyTrainer(Trainer):
    def __init__(
        self,
        fake_max_steps_for_scheduler: Optional[int] = None,
        fake_max_epochs_for_scheduler: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        trainer_module = inspect.getmodule(Trainer)
        trainer_module.CheckpointConnector = MyCheckpointConnector  # Override to save metrics even if not in fault-tolerant mode + don't save weights with trainer state unless after full epoch
        trainer_module.SignalConnector = MySignalConnector  # Override to handle slurm preemption
        super().__init__(*args, **kwargs)
        self._fake_max_steps_for_scheduler = fake_max_steps_for_scheduler
        self._fake_max_epochs_for_scheduler = fake_max_epochs_for_scheduler

        # Fix forced shuffling for DDP
        self._data_connector._resolve_sampler = types.MethodType(
            _wrap_resolve_sampler(self._data_connector._resolve_sampler), self._data_connector
        )

    def get_logger(self, logger_klass):
        loggers = self.logger
        if not isinstance(loggers, LoggerCollection):
            loggers = [loggers]
        requested_logger = None
        for logger_ in loggers:
            if isinstance(logger_, logger_klass):
                requested_logger = logger_
                break
        return requested_logger

    def upload_files(self, files_to_upload: List[str]):
        wandb_logger = self.get_logger(WandbLogger)
        if wandb_logger is not None and wandb_logger._experiment is not None:
            for file_to_upload in files_to_upload:
                logger.info(f"Saving {file_to_upload}")
                wandb_logger.experiment.save(file_to_upload)

    # Copied as is from trainer.py, just modifying max_steps according to self._fake_max_steps_for_scheduler
    @property
    def estimated_stepping_batches(self) -> Union[int, float]:
        r"""
        Estimated stepping batches for the complete training inferred from DataLoaders, gradient
        accumulation factor and distributed setup.

        Examples::

            def configure_optimizers(self):
                optimizer = ...
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=1e-3, total_steps=self.trainer.estimated_stepping_batches
                )
                return [optimizer], [scheduler]

        """
        max_steps = self.max_steps
        if self._fake_max_steps_for_scheduler is not None:
            max_steps = self._fake_max_steps_for_scheduler

        max_epochs = self.max_epochs
        if self._fake_max_epochs_for_scheduler is not None:
            max_epochs = self._fake_max_epochs_for_scheduler

        accumulation_scheduler = self.accumulation_scheduler

        if accumulation_scheduler.epochs != [0]:
            raise MisconfigurationException(
                "Estimated stepping batches cannot be computed with different"
                " `accumulate_grad_batches` at different epochs."
            )

        # infinite training
        if max_epochs == -1 and max_steps == -1:
            return float("inf")

        if self.train_dataloader is None:
            rank_zero_info("Loading `train_dataloader` to estimate number of stepping batches.")
            self.reset_train_dataloader()

        total_batches = self.num_training_batches

        # iterable dataset
        if total_batches == float("inf"):
            return max_steps

        self.accumulate_grad_batches = accumulation_scheduler.get_accumulate_grad_batches(self.current_epoch)
        effective_batch_size = self.accumulate_grad_batches
        max_estimated_steps = math.ceil(total_batches / effective_batch_size) * max(max_epochs, 1)

        max_estimated_steps = min(max_estimated_steps, max_steps) if max_steps != -1 else max_estimated_steps
        return max_estimated_steps
