from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Any, List, Union
from collections.abc import Mapping
from collections import defaultdict
import os
import shutil
import json
import builtins
import re

import torch
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.trainer.states import TrainerFn

from src.data import STAGE_TO_SPLIT, DatasetHelper

MAP_TO = "->"
LEFT_SIDE_REGEX = re.compile(r"(?:^(\S+)(?:\()(\S+)(?:\))$)|(^[^\(\)\s]+$)")  # cast(key) or key
IDENTITY_FUNCTION = lambda x: x


def _final_key_to_tuple(k):
    if MAP_TO in k:
        left, right = [part.strip() for part in k.split(MAP_TO)]
        casting_op, casted_source_key, source_key = LEFT_SIDE_REGEX.search(left).groups()
        if casting_op is not None and casted_source_key is not None:
            source_key = casted_source_key
            source_cast = getattr(builtins, casting_op)
        elif source_key is not None:
            source_cast = IDENTITY_FUNCTION
        target_key = right.strip()
        return (source_key, target_key, source_cast)
    else:
        return (k, k, IDENTITY_FUNCTION)


@dataclass
class JsonPredictionsSettings:
    raw_example_keys: List[str] = field(default_factory=list)
    preprocessed_example_keys: List[str] = field(default_factory=list)
    interim_batch_keys: List[str] = field(default_factory=list)
    interim_output_keys: List[str] = field(default_factory=list)
    batch_keys: List[str] = field(default_factory=list)
    output_keys: List[str] = field(default_factory=list)
    output_list: bool = False
    enabled: bool = True

    def __post_init__(self):
        if not self.enabled:
            return

        for k in self.batch_keys:
            k = _final_key_to_tuple(k)[0]
            if k not in self.interim_batch_keys:
                self.interim_batch_keys.append(k)

        for k in self.output_keys:
            k = _final_key_to_tuple(k)[0]
            if k not in self.interim_output_keys:
                self.interim_output_keys.append(k)

        self.raw_example_keys = list(map(_final_key_to_tuple, self.raw_example_keys))
        self.preprocessed_example_keys = list(map(_final_key_to_tuple, self.preprocessed_example_keys))
        self.batch_keys = list(map(_final_key_to_tuple, self.batch_keys))
        self.output_keys = list(map(_final_key_to_tuple, self.output_keys))

        if (
            len(self.raw_example_keys) == 0
            and len(self.preprocessed_example_keys) == 0
            and len(self.batch_keys) == 0
            and len(self.output_keys) == 0
        ):
            raise ValueError("No keys provided for writing predictions")


class JsonPredictionsWriter(Callback):
    def __init__(
        self,
        settings: Union[JsonPredictionsSettings, Dict[str, JsonPredictionsSettings]],
    ):
        super().__init__()

        if isinstance(settings, JsonPredictionsSettings):
            self._settings = {None: settings}
        elif all(isinstance(s, JsonPredictionsSettings) for s in settings.values()):
            self._settings = settings
        else:
            raise ValueError("settings must be either a JsonPredictionsSettings or a dict of JsonPredictionsSettings")

        self._predictions_memory = None

        self._temp_dir = None

    # epoch start
    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._epoch_start(trainer, pl_module)

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._epoch_start(trainer, pl_module)

    def _epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        use_barrier = torch.distributed.is_available() and torch.distributed.is_initialized()

        self._temp_dir = os.path.join(trainer.default_root_dir, "temp")

        if trainer.is_global_zero:
            os.makedirs(self._temp_dir, exist_ok=True)
        if use_barrier:
            torch.distributed.barrier()

        self._reset_predictions_memory()

    # batch end
    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        self._batch_end(
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            dataloader_idx,
        )

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        self._batch_end(
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            dataloader_idx,
        )

    def _batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        batch_predictions_to_hierarchical_memory(
            pl_module=pl_module,
            batch=batch,
            outputs=outputs,
            predictions_memory=self._predictions_memory,
            settings=self._settings,
        )

    # epoch end
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._epoch_end(trainer, pl_module)

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._epoch_end(trainer, pl_module)

    def _epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        use_barrier = torch.distributed.is_available() and torch.distributed.is_initialized()

        self._rank_write_on_epoch_end(trainer, pl_module)
        if use_barrier:
            torch.distributed.barrier()
        self._reset_predictions_memory()

        self._write_on_epoch_end(trainer, pl_module)
        if use_barrier:
            torch.distributed.barrier()

        if trainer.is_global_zero:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        if use_barrier:
            torch.distributed.barrier()

    # actual work
    def _reset_predictions_memory(self):
        self._predictions_memory = {
            source_container_name: defaultdict(dict) for source_container_name in self._settings
        }

    def _finalize_predictions(
        self,
        source_container_name: Optional[str],
        all_predictions: Dict[str, Dict[str, Dict[str, Any]]],
        select_prediction_pidx_fn: Callable[[Dict[str, Dict[str, Any]]], int],
        dataset_helper: DatasetHelper,
    ):
        predictions = {}
        raw_dataset = dataset_helper.get_raw_dataset(source_container_name)
        preprocessed_dataset = dataset_helper.preprocessed_dataset

        settings = self._settings[source_container_name]
        batch_keys = settings.batch_keys
        output_keys = settings.output_keys
        raw_example_keys = settings.raw_example_keys
        preprocessed_example_keys = settings.preprocessed_example_keys

        for idx, pidx_to_prediction in all_predictions.items():
            pidx = select_prediction_pidx_fn(pidx_to_prediction)
            prediction = pidx_to_prediction[pidx]
            prediction = {
                **{k[0]: prediction[k[0]] for k in batch_keys if k[0] in prediction},
                **{k[0]: prediction[k[0]] for k in output_keys if k[0] in prediction},
                **{k[0]: raw_dataset[int(idx)][k[0]] for k in raw_example_keys if k[0] in raw_dataset.column_names},
                **{
                    k[0]: preprocessed_dataset[int(pidx)][k[0]]
                    for k in preprocessed_example_keys
                    if k[0] in preprocessed_dataset.column_names
                },
            }

            assert len(prediction.keys()) > 0, (
                "A prediction has no keys. "
                " Make sure you specified the model output and batch_keys/output_keys correctly."
            )

            if len(prediction.keys()) == 1:
                # a prediction with a single key can be unwrapped from the dictionary
                prediction = next(iter(prediction.values()))

            predictions[raw_dataset[int(idx)]["id"]] = prediction

        return predictions

    def _rank_write_on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for source_container_name, predictions_memory in self._predictions_memory.items():
            if not self._settings[source_container_name].enabled:
                continue

            rank_predictions_file_path = os.path.join(
                self._temp_dir, _get_rank_predictions_file_name(pl_module.global_rank, source_container_name)
            )
            with open(rank_predictions_file_path, mode="w") as f:
                json.dump(predictions_memory, f)

    def _unify_rank_predictions(self, world_size: int):
        all_predictions_per_source = {}
        for source_container_name in self._settings.keys():
            if not self._settings[source_container_name].enabled:
                continue

            predictions = defaultdict(dict)
            for rank in range(world_size):
                rank_predictions_file_path = os.path.join(
                    self._temp_dir, _get_rank_predictions_file_name(rank, source_container_name)
                )
                with open(rank_predictions_file_path, mode="r") as f:
                    rank_predictions = json.load(f)

                for idx_str, pidx_to_prediction in rank_predictions.items():
                    predictions[idx_str].update(pidx_to_prediction)
            all_predictions_per_source[source_container_name] = predictions
        return all_predictions_per_source

    def _write_on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        select_prediction_pidx_fn = getattr(
            pl_module, "select_prediction_pidx", JsonPredictionsWriter.select_prediction_pidx
        )

        # TODO: Use more efficient sync with https://github.com/facebookresearch/DPR/blob/02e6454d3217db322c8d9f4401299684b9299723/dpr/utils/dist_utils.py#L36-L96?
        all_predictions_per_source = self._unify_rank_predictions(trainer.world_size)

        dataset_helper = pl_module.get_current_dataset_helper()

        pl_module.predictions = {}
        for source_container_name in self._settings.keys():
            settings = self._settings[source_container_name]
            if not settings.enabled:
                continue

            batch_keys = settings.batch_keys
            output_keys = settings.output_keys
            raw_example_keys = settings.raw_example_keys
            preprocessed_example_keys = settings.preprocessed_example_keys
            output_list = settings.output_list

            predictions = self._finalize_predictions(
                source_container_name,
                all_predictions_per_source[source_container_name],
                select_prediction_pidx_fn,
                dataset_helper,
            )
            pl_module.predictions[source_container_name] = predictions

            if trainer.is_global_zero:
                if not trainer.sanity_checking:
                    split = STAGE_TO_SPLIT[trainer.state.stage]
                    predictions_file_path = f"{split}_predictions.json"
                    if source_container_name is not None:
                        predictions_file_path = f"{source_container_name}_{predictions_file_path}"

                    if trainer.state.fn == TrainerFn.FITTING:
                        checkpoint_callback = trainer.checkpoint_callback
                        if checkpoint_callback is not None:
                            metrics = checkpoint_callback._monitor_candidates(
                                trainer, epoch=trainer.current_epoch, step=trainer.global_step
                            )
                            formatted_name = checkpoint_callback._format_checkpoint_name(
                                checkpoint_callback.filename,
                                metrics,
                                auto_insert_metric_name=checkpoint_callback.auto_insert_metric_name,
                            )
                        else:
                            formatted_name = f"epoch_{trainer.current_epoch}"
                            if trainer.val_check_interval != 1.0:
                                formatted_name += f"_step_{trainer.global_step}"

                        predictions_file_path = predictions_file_path.replace(".json", f"_{formatted_name}.json")

                    predictions = {
                        key: {
                            **{k[1]: k[2](prediction[k[0]]) for k in batch_keys if k[0] in prediction},
                            **{k[1]: k[2](prediction[k[0]]) for k in output_keys if k[0] in prediction},
                            **{k[1]: k[2](prediction[k[0]]) for k in raw_example_keys if k[0] in prediction},
                            **{k[1]: k[2](prediction[k[0]]) for k in preprocessed_example_keys if k[0] in prediction},
                        }
                        if isinstance(prediction, Mapping)
                        else prediction
                        for key, prediction in predictions.items()
                    }
                    with open(predictions_file_path, mode="w") as f:
                        if output_list:
                            json.dump(list(predictions.values()), f)
                        else:
                            json.dump(predictions, f)

                    files_to_upload = [predictions_file_path]
                    trainer.upload_files(files_to_upload)

    @staticmethod
    def select_prediction_pidx(predictions: Dict[str, Dict[str, Any]]) -> int:
        if len(predictions) != 1:
            print(predictions)
        assert len(predictions) == 1, (
            "An example has more than one prediction. "
            "Override JsonPredictionsWriter.select_prediction_pidx in the model with logic to choose the best one."
        )
        return next(iter(predictions.keys()))


def batch_predictions_to_hierarchical_memory(
    pl_module: "pl.LightningModule",
    batch,
    outputs,
    predictions_memory,
    settings: Dict[str, JsonPredictionsSettings],
):
    # DistributedSampler will yield the same instances more than once when batch padding is required,
    # so we need to remember the source of the prediction as well
    batch_size = pl_module.find_batch_size(batch)
    for i in range(batch_size):
        source_container_name = batch["source_container_name"][i] if "source_container_name" in batch else None
        current_settings = settings[source_container_name]
        if not current_settings.enabled:
            continue

        interim_batch_keys = current_settings.interim_batch_keys
        interim_output_keys = current_settings.interim_output_keys

        prediction = {
            **{
                k: batch[k][i]
                for k in (interim_batch_keys if interim_batch_keys is not None else batch.keys())
                if k not in ["tensor_keys", "key_mapping"]
            },
            **{k: outputs[k][i] for k in (interim_output_keys if interim_output_keys is not None else outputs.keys())},
        }

        pidx = batch["pidx"][i]
        idx = pl_module.get_current_dataset_helper().pidx_to_idx(batch["pidx"][i])
        pidx_to_prediction = predictions_memory[source_container_name][str(idx)]
        pidx_to_prediction[str(pidx)] = prediction


def _get_rank_predictions_file_name(rank: int, source_container_name: Optional[str]) -> str:
    file_name = f"predictions_{rank}.json"
    if source_container_name is not None:
        file_name = f"{source_container_name}_{file_name}"
    return file_name
