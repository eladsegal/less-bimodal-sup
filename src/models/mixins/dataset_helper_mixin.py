from typing import Any

from src.pl import MyLightningModule
from src.data import STAGE_TO_SPLIT, DatasetHelper
from src.data.datamodules.base_datamodule import BaseDataModule

DEFAULT = "default"


class DatasetHelperMixin(MyLightningModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_helpers = {}

    def _set_dataset_helper(self, dataset_helper: DatasetHelper = None) -> None:
        if dataset_helper is not None:
            self.dataset_helpers[DEFAULT] = dataset_helper
            return
        split = STAGE_TO_SPLIT[self.trainer.state.stage]
        if self.dataset_helpers.get(split) is None:
            datamodule = self.trainer.datamodule
            if datamodule is not None:
                self.dataset_helpers[split] = DatasetHelper(
                    split=split,
                    dataset_container=datamodule.dataset_container,
                    preprocessed_dataset=datamodule.preprocessed_datasets[split],
                )

    def get_current_dataset_helper(self) -> DatasetHelper:
        if self.trainer is None:
            return self.dataset_helpers.get(DEFAULT)
        split = STAGE_TO_SPLIT[self.trainer.state.stage]
        if self.dataset_helpers.get(split) is None:
            return None
        return self.dataset_helpers[split]

    def on_train_start(self) -> None:
        super().on_train_start()
        self._set_dataset_helper()

    def on_validation_start(self) -> None:
        super().on_validation_start()
        self._set_dataset_helper()

    def on_test_start(self) -> None:
        super().on_test_start()
        self._set_dataset_helper()
