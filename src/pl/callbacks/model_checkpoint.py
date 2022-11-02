from typing import Optional, Any, Dict, List
import json
import csv
import os
import torch
import time
import datetime
import logging
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.types import _METRIC

logger = logging.getLogger(__name__)


class BetterModelCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        dirpath: Optional[str] = "./checkpoints",
        auto_insert_metric_name: bool = False,
        save_top_k: int = 1,
        save_last: bool = True,
        save_best_weight_only: bool = True,
        save_on_train_epoch_end: bool = False,
        save_also_top_k_weight_only: int = 2,
        specific_weights_to_save: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            dirpath=dirpath,
            auto_insert_metric_name=auto_insert_metric_name,
            save_top_k=save_top_k,
            save_last=save_last,
            save_on_train_epoch_end=save_on_train_epoch_end,
            **kwargs,
        )
        assert (
            save_top_k >= 1 or save_top_k == -1
        ), "save_top_k must be >= 1 or -1 because BetterModelCheckpoint relies on it"
        self._save_best_weight_only = save_best_weight_only
        self._save_also_top_k_weight_only = save_also_top_k_weight_only
        self._specific_weights_to_save = specific_weights_to_save
        self._best_k_models_weight_only = {} if self._save_also_top_k_weight_only != 0 else None
        self._best_model_path = None

        self._training_start_time = 0
        self._val_count = 0

    def on_save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> Dict[str, Any]:
        state = super().on_save_checkpoint(trainer, pl_module, checkpoint)
        state.update(
            {
                "training_time": time.time() - self._training_start_time,
                "_best_k_models_weight_only": self._best_k_models_weight_only,
                "val_count": self._val_count + 1,
            }
        )
        return state

    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", callback_state: Dict[str, Any]
    ) -> None:
        super().on_load_checkpoint(trainer, pl_module, callback_state)
        self._training_start_time = -callback_state.get("training_time", 0)
        self._best_k_models_weight_only = callback_state.get("_best_k_models_weight_only")
        self._best_model_path = callback_state.get("best_model_path")  # it is intentional not using _best_model_path
        self._val_count = callback_state.get("val_count")

    def _monitor_candidates(self, trainer: "pl.Trainer", epoch: int, step: int) -> Dict[str, _METRIC]:
        monitor_candidates = super()._monitor_candidates(trainer, epoch, step)
        monitor_candidates["i"] = self._val_count
        return monitor_candidates

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_start(trainer, pl_module)
        self._training_start_time += time.time()

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_validation_start(trainer, pl_module)

        if trainer.state.fn != TrainerFn.FITTING:
            return

        monitor_candidates = self._monitor_candidates(trainer, epoch=trainer.current_epoch, step=trainer.global_step)
        filepath = self.format_checkpoint_name(monitor_candidates)

        if not self.save_weights_only and self._save_also_top_k_weight_only != 0:
            # Save weights only in addition to a full checkpoint
            self._best_k_models_weight_only[self._training_checkpoint_path_to_weights_only(filepath)] = None
            weights_only_filepath = self._training_checkpoint_path_to_weights_only(filepath)
            trainer.save_checkpoint(weights_only_filepath, weights_only=True)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.state.fn == TrainerFn.FITTING:
            if not self.save_weights_only and self._save_also_top_k_weight_only != 0:
                # Save weights only in addition to a full checkpoint - handle top k
                monitor_candidates = self._monitor_candidates(
                    trainer, epoch=trainer.current_epoch, step=trainer.global_step
                )
                for path, score in self._best_k_models_weight_only.items():
                    if score is None:
                        self._best_k_models_weight_only[path] = monitor_candidates.get(self.monitor)
                        break

        super().on_validation_end(trainer, pl_module)
        self._save_file_metrics(trainer)

        if trainer.state.fn == TrainerFn.FITTING:
            if self._save_best_weight_only:
                is_new_best = self._best_model_path != self.best_model_path
                if is_new_best:
                    trainer.save_checkpoint(
                        os.path.join(os.path.dirname(self.best_model_path), "best.ckpt"), weights_only=True
                    )

            if not self.save_weights_only and self._save_also_top_k_weight_only != 0:
                k = (
                    len(self._best_k_models_weight_only) + 1
                    if self._save_also_top_k_weight_only == -1
                    else self._save_also_top_k_weight_only
                )
                if len(self._best_k_models_weight_only) > k and k > 0:
                    monitor_op = {"min": max, "max": min}[self.mode]  # Reversed because we want the worst
                    del_filepath = monitor_op(self._best_k_models_weight_only, key=self._best_k_models_weight_only.get)
                    self._best_k_models_weight_only.pop(del_filepath)
                    if del_filepath is not None:
                        if (
                            self._specific_weights_to_save is None
                            or Path(del_filepath).stem not in self._specific_weights_to_save
                        ):
                            trainer.training_type_plugin.remove_checkpoint(del_filepath)

            self._best_model_path = self.best_model_path
            self._val_count += 1

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.on_validation_end(trainer, pl_module)

    @rank_zero_only
    def _save_file_metrics(self, trainer: "pl.Trainer"):
        os.makedirs(self.dirpath, exist_ok=True)

        # Prepare metrics
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in trainer.logged_metrics.items()}

        if trainer.state.fn == TrainerFn.FITTING:
            metrics["epoch"] = trainer.current_epoch
            metrics["step"] = trainer.global_step
            metrics["i"] = self._val_count

            training_elapsed_time = time.time() - self._training_start_time
            metrics["training_time"] = str(datetime.timedelta(seconds=training_elapsed_time))

            if self.monitor is not None:
                best_metrics_path = os.path.join(self.dirpath, "metrics_best.json")
                # If needed, save new best metrics
                is_new_best = self._best_model_path != self.best_model_path
                if is_new_best:
                    with open(best_metrics_path, mode="w") as f:
                        json.dump(metrics, f, indent=4)

        # Add metrics to csv
        csv_path = os.path.join(self.dirpath, "metrics.csv")
        csv_exits = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys(), delimiter=",")
            if not csv_exits:
                writer.writeheader()
            writer.writerow(metrics)

        if trainer.state.fn == TrainerFn.FITTING:
            # Load best metrics
            if os.path.exists(best_metrics_path):
                with open(best_metrics_path, mode="r") as f:
                    best_metrics = json.load(f)

                # Update metrics with best
                metrics.update({f"best_{k}": v for k, v in best_metrics.items()})

        # Save metrics for the current validation run
        template_filename = self.filename.replace("training_", "")
        if trainer.state.fn != TrainerFn.FITTING:
            for key in ["step", "epoch"]:
                if key not in metrics:
                    metrics[key] = ""
        logger.info(
            json.dumps(
                {k: v for k, v in metrics.items() if any(k.startswith(prefix) for prefix in ["val/", "test/"])},
                indent=4,
            )
        )
        metrics_filename = f"metrics_{template_filename.format(**metrics)}.json"
        file_path = os.path.join(self.dirpath, metrics_filename)
        with open(file_path, mode="w") as f:
            json.dump(metrics, f, indent=4)

        files_to_upload = []
        files_to_upload.append(csv_path)
        files_to_upload.append(file_path)
        if trainer.state.fn == TrainerFn.FITTING:
            self.to_yaml()
            best_k_models = os.path.join(self.dirpath, "best_k_models.yaml")
            if os.path.isfile(best_k_models):
                files_to_upload.append(best_k_models)
            if is_new_best:
                files_to_upload.append(best_metrics_path)
        trainer.upload_files(files_to_upload)

    @staticmethod
    def _training_checkpoint_path_to_weights_only(filepath):
        weights_only_filepath = os.path.join(
            os.path.dirname(filepath), os.path.basename(filepath).replace("training", "model")
        )
        if weights_only_filepath == filepath:
            weights_only_filepath = os.path.join(
                os.path.dirname(filepath), "weights_only_" + os.path.basename(filepath)
            )
        return weights_only_filepath
