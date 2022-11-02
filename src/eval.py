# MANDATORY - START
from typing import Type, Union
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

from pathlib import Path
import logging
from omegaconf import DictConfig, open_dict
import hydra
from hydra.utils import instantiate
from hydra._internal.utils import _locate
import torch
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.base import DummyExperiment

from src.pl import MyTrainer
from src.utils.loggers import instantiate_loggers
from src.data.dataset_containers import BaseDatasetContainer, ComplexDatasetContainer, MultiDatasetContainer
from src.data.datamodules.base_datamodule import BaseDataModule
from src.models.base_model import BaseModel
from src.utils.execution_info import execution_info_config_modifications
from src.evaluators import Evaluator, HfEvaluator

logger = logging.getLogger(__name__)

# MANDATORY - START
random_temp_name = get_random_chars(prefix="temp_")


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
    )  # model_dict cannot be passed in the constructor due to hydra issues

    if cfg["global"].get("only_preprocess_data", False):
        logger.info("Data preprocessing completed")
        exit()

    evaluator: Evaluator = (
        HfEvaluator(instantiate(cfg.metrics, _convert_="all")) if cfg["global"].get("use_evaluator", True) else None
    )
    if ckpt_path is not None:
        with open_dict(cfg.model):
            model_target = cfg.model.pop("_target_")  # Removing _target_ so `instantiate` won't create the model
        model_class: Type[BaseModel] = _locate(model_target)
        model = model_class.load_from_checkpoint(
            ckpt_path,
            **instantiate(cfg.model, evaluator=evaluator, _convert_="all"),
            cfg=cfg_saveable,
        )  # **dm.model_kwargs shouldn't be passed as the right values should be in the checkpoint
    else:
        model = instantiate(cfg.model, **dm.model_kwargs, evaluator=evaluator, cfg=cfg_saveable, _convert_="all")
        if manually_load_weights:
            model.manually_load_weights(model_dict)

    # Run validate/test
    getattr(trainer, Path(sys.argv[0]).stem)(model, dm)
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
