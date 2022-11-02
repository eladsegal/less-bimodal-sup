from typing import Union, Dict, Any
from argparse import Namespace

import os
import sys
import shlex
from copy import deepcopy

from hydra.utils import get_original_cwd

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.logger import _convert_params, _flatten_dict, _sanitize_callable_params

try:
    import wandb
    from wandb.wandb_run import Run
except ImportError:
    # needed for test mocks, these tests shall be updated
    wandb, Run = None, None

from utils.general import smart_open

import logging

logger = logging.getLogger(__name__)


class BetterWandbLogger(WandbLogger):
    def __init__(self, **kwargs):
        if "dir" not in kwargs:
            kwargs["dir"] = os.path.abspath(".")
        self._command = kwargs.pop("command", None)
        super().__init__(**kwargs)

    @property
    @rank_zero_experiment
    def experiment(self) -> Run:
        is_new_experiment = self._experiment is None

        temp_cwd = os.getcwd()
        os.chdir(get_original_cwd())
        prev_argv = sys.argv
        if self._command is not None:
            command_split = shlex.split(self._command)
            sys.argv = ["invisible_part"] + command_split[2:]
        experiment = super().experiment
        sys.argv = prev_argv
        os.chdir(temp_cwd)

        if is_new_experiment:
            self._experiment.define_metric("train/n_examples_overall")
            self._experiment.define_metric("epoch")
            self._experiment.define_metric("train/*", step_metric="train/n_examples_overall")
            self._experiment.define_metric("val/*", step_metric="epoch")

            metadata_path = os.path.abspath("metadata")

            with smart_open(os.path.join(metadata_path, "wandb_id"), mode="w") as f:
                f.write(experiment.id)

            from src.utils.reproducibility import IMPORTANT_FILE_PATHS  # TODO: This is bad

            for file_path in IMPORTANT_FILE_PATHS:
                if os.path.isfile(file_path):
                    experiment.save(file_path)

        return experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = _convert_params(deepcopy(params))

        # keep only cfg from params for wandb
        params = params["cfg"] if "cfg" in params else params
        for env_var in ["CUDA_VISIBLE_DEVICES", "PL_FAULT_TOLERANT_TRAINING"]:
            if env_var in os.environ:
                params[env_var] = os.environ[env_var]

        params = _flatten_dict(params)
        params = _sanitize_callable_params(params)
        self.experiment.config.update(params, allow_val_change=True)
