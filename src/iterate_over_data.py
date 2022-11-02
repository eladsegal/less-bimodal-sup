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
from src.utils.config import cfg_save_preparation
from src.utils.logging import fix_logging
from src.utils.reproducibility import reproducibility_init
from utils.gpu import config_modifications_for_gpu

# MANDATORY - END

import logging
import traceback
from omegaconf import DictConfig, open_dict
import hydra
from hydra.utils import instantiate
from pytorch_lightning.loggers import WandbLogger

from src.utils.loggers import instantiate_loggers
from src.data.dataset_containers import BaseDatasetContainer, ComplexDatasetContainer, MultiDatasetContainer
from src.data.datamodules.base_datamodule import BaseDataModule
from src.pl.callbacks.progress import tqdm
from src.utils.execution_info import execution_info_config_modifications

logger = logging.getLogger(__name__)

# MANDATORY - START
random_temp_name = get_random_chars(prefix="temp_")


# using the hydra wrapper only for logging, the creation of the output directory and changing the workdir
@hydra.main(config_path="../configs/templates", config_name=random_temp_name)
def main(cfg: DictConfig) -> None:
    sys.argv = original_argv
    fix_logging()

    execution_info_config_modifications(cfg)

    with open_dict(cfg):
        cfg.trainer.gpus = 0
    config_modifications_for_gpu(cfg)

    reproducibility_init(cfg)
    cfg_saveable = cfg_save_preparation(cfg)
    # MANDATORY - END

    loggers = instantiate_loggers(cfg, cfg_saveable)

    dataset_container: Union[BaseDatasetContainer, ComplexDatasetContainer, MultiDatasetContainer] = instantiate(
        cfg.dataset_container, _convert_="all"
    )
    dm: BaseDataModule = instantiate(cfg.datamodule, dataset_container=dataset_container, _convert_="all")

    # Using it manually so the model could use dm.model_kwargs for init
    dm.prepare_data()  # TODO: Should be fixed to be called with the same conditions as in data_connector.DataConnector.prepare_data
    dm.setup()

    if cfg["global"].get("only_preprocess_data", False):
        logger.info("Data preprocessing completed")
        exit()

    for dataloader_fn in ["val_dataloader", "test_dataloader", "train_dataloader"]:
        logger.info(f"Iterating through {dataloader_fn}")
        try:
            dataloader = getattr(dm, dataloader_fn)()
            if dataloader is not None:
                for batch in tqdm(dataloader):
                    continue
        except Exception:
            logger.info(f"Failed iteration through {dataloader_fn} with exception:")
            traceback.print_exc()

    for metrics_logger in loggers:
        if isinstance(metrics_logger, WandbLogger):
            metrics_logger.experiment.config.update({"metadata/status": "completed"}, allow_val_change=True)


if __name__ == "__main__":
    # MANDATORY - START
    global original_argv
    __main__.__file__ = sys.argv[0]  # To make sure program name is correct in wandb
    __main__.__spec__ = None  # For self-executing DDP
    original_argv = hydra_prep_with_multiprocessing_support(random_temp_name)

    main()
    # MANDATORY - END
