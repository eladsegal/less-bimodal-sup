from typing import Dict, Any

import os

from pytorch_lightning import LightningModule

from src.utils.inspection import get_fqn

import logging

logger = logging.getLogger(__name__)


class MyLightningModule(LightningModule):
    def __init__(self, *args: Any, cfg: Dict[str, Any] = None, ignore_saving=None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if cfg is not None:
            self._cfg = cfg
            self.save_hyperparameters(ignore=ignore_saving)
        else:
            self._cfg = None

        # TODO: Understand if this class is really needed in addition to BaseModel, and define its roles if so

    def on_before_manually_load_weights(self, checkpoint: Dict[str, Any]) -> None:
        pass

    def manually_load_weights(self, checkpoint: Dict[str, Any]):
        self.on_before_manually_load_weights(checkpoint)
        state_dict = checkpoint["state_dict"]

        for k in list(state_dict.keys()):
            if k.startswith("_metrics_dict."):
                del state_dict[k]

        SUPPORTED_PAIRS = [
            (
                "src.models.vl.pretraining_model.PretrainingModel",
                "src.models.vl.classifier_model.ClassifierModel",
            ),
        ]
        if checkpoint["hyper_parameters"]["cfg"]["model"]["__target__"] != get_fqn(type(self)):
            pair = (checkpoint["hyper_parameters"]["cfg"]["model"]["__target__"], get_fqn(type(self)))
            if pair not in SUPPORTED_PAIRS:
                raise ValueError(f"Unsupported model pair {pair} for weight loading")

        target_state_dict_keys = list(self.state_dict().keys())
        for k in list(state_dict.keys()):
            if k not in target_state_dict_keys:
                logging.info(f"Removing key {k} from state_dict")
                del state_dict[k]

        try:
            self.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            logger.error(f"Failed with load_state_dict(strict=True):\n{e}")
            self.load_state_dict(state_dict, strict=False)

    @staticmethod
    def add_to_ignore_saving_in_kwargs(name: str, kwargs: Dict[str, Any]):
        if "ignore_saving" not in kwargs:
            kwargs["ignore_saving"] = []
        kwargs["ignore_saving"].append(name)
