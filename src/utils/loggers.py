from typing import Dict, Any
from collections.abc import Sequence
import os

from omegaconf import DictConfig, open_dict
from hydra.utils import instantiate
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from src.utils.logging import redirect_stdout

# import wandb


def instantiate_loggers(cfg: DictConfig, cfg_saveable: Dict[str, Any] = None):
    loggers = []
    logger_configs = cfg.trainer.get("logger", [])
    if not isinstance(logger_configs, bool):
        if not isinstance(logger_configs, Sequence):
            logger_configs = [logger_configs]

        for logger_config in logger_configs:
            kwargs = {}

            if "wandb" in logger_config._target_.lower():
                metadata_path = os.path.abspath("metadata")
                wandb_id_path = os.path.join(metadata_path, "wandb_id")
                if os.path.exists(wandb_id_path):
                    with open(wandb_id_path, mode="r") as f:
                        wandb_id = f.read()
                    kwargs["id"] = wandb_id
                # kwargs["settings"] = wandb.Settings(start_method="fork")

            metrics_logger = instantiate(logger_config, **kwargs, _convert_="all")
            metrics_logger.experiment
            metrics_logger.log_hyperparams({"cfg": cfg_saveable})
            loggers.append(metrics_logger)

        with open_dict(cfg):
            del cfg.trainer.logger

        if rank_zero_only.rank == 0:
            wandb_logger = None
            wandb_index = -1
            for i, logger in enumerate(loggers):
                if isinstance(logger, WandbLogger):
                    wandb_logger = logger
                    wandb_index = i
                    break

            if wandb_logger is not None:
                experiment_id = wandb_logger.experiment.id
                if cfg_saveable is not None:
                    cfg_saveable["trainer"]["logger"][wandb_index]["id"] = experiment_id

    redirect_stdout()
    return loggers
