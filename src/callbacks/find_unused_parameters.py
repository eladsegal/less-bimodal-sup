import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)


class FindUnusedParameters(Callback):
    def on_after_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logger.info("Unused parameters:")
        for name, param in pl_module.named_parameters():
            if param.grad is None:
                logger.info(name)
