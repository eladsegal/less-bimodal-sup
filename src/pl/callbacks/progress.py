from typing import Any, Dict, Union, Optional

import sys
from time import time

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar, _update_n, convert_inf

# In TQDMProgressBar auto.tqdm is used as _tqdm and when it is used
# to iterate over a dataloader it results in a duplication of every worker.
# Therefore we make sure to use the regular tqdm here
from tqdm import tqdm as _tqdm

import logging

logger = logging.getLogger(__name__)
tqdm_logger = logging.getLogger("tqdm")
tqdm_logger.propagate = False


def replace_cr_with_newline(message: str) -> str:
    """
    TQDM and requests use carriage returns to get the training line to update for each batch
    without adding more lines to the terminal output. Displaying those in a file won't work
    correctly, so we'll just make sure that each batch shows up on its one line.
    """
    # In addition to carriage returns, nested progress-bars will contain extra new-line
    # characters and this special control sequence which tells the terminal to move the
    # cursor one line up.
    message = message.replace("\r", "").replace("\n", "").replace("[A", "")
    if message and message[-1] != "\n":
        message += "\n"
    return message


class TqdmToLogsWriter(object):
    def __init__(self):
        self.last_message_written_time = 0.0

    def write(self, message):
        file_friendly_message: str = None
        sys.stderr.write(message)

        # Every 10 seconds we also log the message.
        now = time()
        if now - self.last_message_written_time >= 10 or "100%" in message:
            if file_friendly_message is None:
                file_friendly_message = replace_cr_with_newline(message)
            for message in file_friendly_message.split("\n"):
                message = message.strip()
                if len(message) > 0:
                    tqdm_logger.info(message)
                    self.last_message_written_time = now

    def flush(self):
        sys.stderr.flush()


class tqdm(_tqdm):
    def __init__(self, *args, **kwargs):
        new_kwargs = {
            "file": TqdmToLogsWriter(),
            **kwargs,
        }
        super().__init__(*args, **new_kwargs)


class BetterTQDMProgressBar(TQDMProgressBar):
    @property
    def total_val_batches(self) -> Union[int, float]:
        """The total number of validation batches, which may change from epoch to epoch.

        Use this to set the total number of iterations in the progress bar. Can return ``inf`` if the validation
        dataloader is of infinite size.
        """
        if self.trainer.sanity_checking:
            return sum(self.trainer.num_sanity_val_batches)
        return sum(self.trainer.num_val_batches)

    def init_train_tqdm(self) -> tqdm:
        """Override this to customize the tqdm bar for training."""
        bar = tqdm(
            desc="Training",
            initial=self.train_batch_idx % self.total_train_batches if self.total_train_batches is not None else 0,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            smoothing=0,
        )
        return bar

    def init_predict_tqdm(self) -> tqdm:
        """Override this to customize the tqdm bar for predicting."""
        bar = tqdm(
            desc="Predicting",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            smoothing=0,
        )
        return bar

    def init_validation_tqdm(self) -> tqdm:
        """Override this to customize the tqdm bar for validation."""
        # The main progress bar doesn't exist in `trainer.validate()`
        bar = tqdm(
            desc="Validating",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
        )
        return bar

    def init_test_tqdm(self) -> tqdm:
        """Override this to customize the tqdm bar for testing."""
        bar = tqdm(
            desc="Testing",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
        )
        return bar

    def on_sanity_check_start(self, *_: Any) -> None:
        self.val_progress_bar = self.init_sanity_tqdm()

    def on_sanity_check_end(self, *_: Any) -> None:
        self.val_progress_bar.close()

    def on_train_start(self, *_: Any) -> None:
        pass

    def on_train_epoch_start(self, trainer: "pl.Trainer", *_: Any) -> None:
        rank_zero_info("Training")
        self.main_progress_bar = self.init_train_tqdm()
        self.main_progress_bar.set_description(f"Training: Epoch {trainer.current_epoch}", refresh=False)
        self.main_progress_bar.total = convert_inf(self.total_train_batches)

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", *_: Any) -> None:
        if self._should_update(self.train_batch_idx, convert_inf(self.total_train_batches)):
            self.main_progress_bar.set_postfix(
                self.get_metrics(trainer, pl_module, prefix_to_keep="train/"), refresh=False
            )
            _update_n(self.main_progress_bar, self.train_batch_idx)

    def on_train_epoch_end(self, *_: Any) -> None:
        self.main_progress_bar.close()

    def on_validation_start(self, trainer: "pl.Trainer", *_: Any) -> None:
        if trainer.sanity_checking:
            self.val_progress_bar.total = sum(trainer.num_sanity_val_batches)
        else:
            rank_zero_info("Validating")
            self.val_progress_bar = self.init_validation_tqdm()
            self.val_progress_bar.set_description(f"Validating: Epoch {trainer.current_epoch}", refresh=False)
            self.val_progress_bar.total = convert_inf(self.total_val_batches)

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", *_: Any) -> None:
        if self._should_update(self.val_batch_idx, convert_inf(self.total_val_batches)):
            self.val_progress_bar.set_postfix(
                self.get_metrics(trainer, pl_module, prefix_to_keep="val/"), refresh=False
            )
            _update_n(self.val_progress_bar, self.val_batch_idx)

    def on_validation_end(self, *_: Any) -> None:
        self.val_progress_bar.close()

    def on_test_start(self, *_: Any) -> None:
        rank_zero_info("Testing")
        self.test_progress_bar = self.init_validation_tqdm()
        self.test_progress_bar.set_description(f"Testing", refresh=False)
        self.test_progress_bar.total = convert_inf(self.total_test_batches)

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", *_: Any) -> None:
        if self._should_update(self.test_batch_idx, convert_inf(self.total_test_batches)):
            self.test_progress_bar.set_postfix(
                self.get_metrics(trainer, pl_module, prefix_to_keep="test/"), refresh=False
            )
            _update_n(self.test_progress_bar, self.test_batch_idx)

    def on_test_end(self, *_: Any) -> None:
        self.test_progress_bar.close()

    def on_predict_epoch_start(self, *_: Any) -> None:
        self.predict_progress_bar = self.init_predict_tqdm()
        self.predict_progress_bar.total = convert_inf(self.total_predict_batches)

    def on_predict_batch_end(self, *_: Any) -> None:
        if self._should_update(self.predict_batch_idx, convert_inf(self.total_predict_batches)):
            _update_n(self.predict_progress_bar, self.predict_batch_idx)

    def on_predict_end(self, *_: Any) -> None:
        self.predict_progress_bar.close()

    def get_metrics(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", prefix_to_keep
    ) -> Dict[str, Union[int, str]]:
        metrics = super().get_metrics(trainer, pl_module)
        if "step" not in metrics:
            metrics["train/step"] = trainer.global_step
        return format_dict(filter_dict(metrics, prefix_to_keep))

    def _should_update(self, idx: int, total: Optional[int] = None) -> bool:
        return (self.refresh_rate > 0 and idx % self.refresh_rate == 0) or (total is not None and idx == total)


def filter_dict(progress_bar_dict, prefix_to_keep):
    return {k: v for k, v in progress_bar_dict.items() if k.startswith(prefix_to_keep)}


def format_dict(progress_bar_dict):
    return {k: str(int(v)) if "n_examples" in k else v for k, v in progress_bar_dict.items()}
