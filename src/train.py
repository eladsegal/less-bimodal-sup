# MANDATORY - START
from typing import Union
import __main__
import sys, os

sys.path.insert(0, os.getcwd())
from src.utils.logging import set_logger

set_logger()
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
from scripts.namegenerator import get_random_chars
from src.utils.config import hydra_prep_with_multiprocessing_support
from utils.gpu import config_modifications_for_gpu
from utils.general import resolve_relative_paths
from src.utils.config import cfg_save_preparation
from src.utils.logging import fix_logging
from src.utils.reproducibility import reproducibility_init

# MANDATORY - END

import logging
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import torch
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.base import DummyExperiment

from src.utils.loggers import instantiate_loggers
from src.data.dataset_containers import BaseDatasetContainer, ComplexDatasetContainer, MultiDatasetContainer
from src.data.datamodules.base_datamodule import BaseDataModule
from src.pl import MyTrainer, MyLightningModule
from src.utils.execution_info import execution_info_config_modifications
from src.evaluators import Evaluator, HfEvaluator

logger = logging.getLogger(__name__)

# MANDATORY - START
random_temp_name = get_random_chars(prefix="temp_")


# using the hydra wrapper only for logging, the creation of the output directory and changing the workdir
@hydra.main(config_path="../configs/templates", config_name=random_temp_name)
def main(cfg: DictConfig) -> None:
    sys.argv = original_argv
    fix_logging()

    execution_info_config_modifications(cfg)
    config_modifications_for_gpu(cfg)

    reproducibility_init(cfg)
    cfg_saveable = cfg_save_preparation(cfg)
    # MANDATORY - END

    loggers = instantiate_loggers(cfg, cfg_saveable)
    trainer: MyTrainer = instantiate(cfg.trainer, logger=loggers, _convert_="all")

    manually_load_weights = False
    model_dict = None
    ckpt_path = resolve_relative_paths(cfg["global"].get("ckpt_path"))
    load_weights = resolve_relative_paths(cfg["global"].get("load_weights"))
    if ckpt_path is not None:
        model_dict = torch.load(ckpt_path, map_location="cpu")
    elif load_weights is not None:
        model_dict = torch.load(load_weights, map_location="cpu")
        manually_load_weights = True

    dataset_container: Union[BaseDatasetContainer, ComplexDatasetContainer, MultiDatasetContainer] = instantiate(
        cfg.dataset_container, _convert_="all"
    )
    dm: BaseDataModule = instantiate(cfg.datamodule, dataset_container=dataset_container, _convert_="all")

    # Using it manually so the model could use dm.model_kwargs for init
    dm.prepare_data()  # TODO: Should be fixed to be called with the same conditions as in data_connector.DataConnector.prepare_data
    dm.setup(
        model_dict=model_dict if not manually_load_weights else None
    )  # Cannot be passed in the constructor due to hydra issues

    if cfg["global"].get("only_preprocess_data", False):
        logger.info("Data preprocessing completed")
        exit()

    evaluator: Evaluator = HfEvaluator(instantiate(cfg.metrics, _convert_="all"))
    model: MyLightningModule = instantiate(
        cfg.model, **dm.model_kwargs, evaluator=evaluator, cfg=cfg_saveable, _convert_="all"
    )
    if manually_load_weights:
        model.manually_load_weights(model_dict)

    if cfg.trainer.get("auto_lr_find", False):
        import pytorch_lightning.tuner.lr_finder as lr_finder_module
        from src.pl.callbacks.progress import tqdm

        lr_finder_module.tqdm = tqdm

        lr_finder = trainer.tuner.lr_find(model, dm, **cfg.get("lr_find_kwargs", {}))
        logger.info(f"Suggested learning rate is {lr_finder.suggestion()}")
        fig = lr_finder.plot(suggest=True)
        fig.savefig("lr_finder.png", bbox_inches="tight")

        import csv

        with open("lr_finder.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["lr", "loss"])
            writer.writerows(
                [lr, loss] for lr, loss in zip(fig.axes[0].lines[0].get_xdata(), fig.axes[0].lines[0].get_ydata())
            )
    else:
        trainer.fit(model, dm, ckpt_path=ckpt_path)
    rank_zero_info(f"Working directory: {os.getcwd()}")
    for metrics_logger in loggers:
        if isinstance(metrics_logger, WandbLogger) and not isinstance(metrics_logger.experiment, DummyExperiment):
            metrics_logger.experiment.config.update({"metadata/status": "completed"}, allow_val_change=True)


if __name__ == "__main__":
    # MANDATORY - START
    global original_argv
    __main__.__file__ = sys.argv[0]  # To make sure program name is correct in wandb
    __main__.__spec__ = None  # For self-executing DDP
    original_argv = hydra_prep_with_multiprocessing_support(random_temp_name)

    main()
    # MANDATORY - END
