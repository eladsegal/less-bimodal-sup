from typing import Any, Dict, Callable
from collections import defaultdict

import torch

from src.pl import MyLightningModule
from src.data.dataset_helper import DatasetHelper
from src.callbacks.json_predictions_writer import batch_predictions_to_hierarchical_memory, JsonPredictionsWriter


class PredictionMixin(MyLightningModule):
    def predict(self, batch: Dict[str, Any], dataset_helper: DatasetHelper, **kwargs):
        with torch.no_grad():
            self._set_dataset_helper(dataset_helper)
            batch = self.transfer_batch_to_device(batch, self.device)
            outputs = self.inference_for_predict(batch)

            predictions_memory = defaultdict(dict)
            batch_predictions_to_hierarchical_memory(
                pl_module=self,
                batch=batch,
                outputs=outputs,
                predictions_memory=predictions_memory,
                interim_batch_keys=["pidx"],
            )
            predictions = {}
            select_prediction_pidx_fn = getattr(
                self, "select_prediction_pidx", JsonPredictionsWriter.select_prediction_pidx
            )
            for idx, pidx_to_prediction in predictions_memory.items():
                pidx = select_prediction_pidx_fn(pidx_to_prediction)
                prediction = pidx_to_prediction[pidx]
                predictions[dataset_helper.raw_dataset[int(idx)]["id"]] = prediction
        return predictions

    def inference_for_predict(self, batch: Dict[str, Any], **kwargs):
        """
        Override this method to implement your own inference logic.
        """
        return self(batch)
