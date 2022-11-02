from typing import Dict, List, Any

from src.models.base_model import BaseModel
from src.metrics import AccuracyByLabels, VqaScoreByLabels

import logging

logger = logging.getLogger(__name__)


class DiscriminativeVQAModel(BaseModel):
    def __init__(
        self,
        ans2label: Dict[str, int],
        **kwargs,
    ):
        DiscriminativeVQAModel.add_to_ignore_saving_in_kwargs("classifier_classes_reorder", kwargs)
        super().__init__(**kwargs)

        self._label2ans = [None for i in range(len(ans2label))]
        for ans, label in ans2label.items():
            self._label2ans[label] = ans

        if self._objective_format in ["cross_entropy"]:
            self.define_metric(["train", "val", "test"], "accuracy_by_labels", AccuracyByLabels)
        elif self._objective_format in ["binary_cross_entropy_with_logits", "volta", "meter"]:
            self.define_metric(["train", "val", "test"], "vqa_score_by_labels", VqaScoreByLabels)

    def on_train_start(self) -> None:
        super().on_train_start()
        if self._objective_format in ["cross_entropy"]:
            self.get_metric("train", "accuracy_by_labels").window_length = self.trainer.log_every_n_steps
        elif self._objective_format in ["binary_cross_entropy_with_logits", "volta", "meter"]:
            self.get_metric("train", "vqa_score_by_labels").window_length = self.trainer.log_every_n_steps

    def enhance_outputs(self, batch: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        outputs["predicted_label"] = outputs["logits"].argmax(-1).tolist()
        outputs["predicted_answer"] = [self._label2ans[label] for label in outputs["predicted_label"]]

    def step_with_metrics(self, batch: Dict[str, Any], batch_idx: int, enhance_outputs=False) -> Dict[str, Any]:
        outputs = self(batch, enhance_outputs=enhance_outputs)
        metric_result = None

        if self._objective_format in ["cross_entropy"]:
            if "labels" in batch:
                accuracy_metric = self.get_current_metric("accuracy_by_labels")
                metric_result = accuracy_metric(outputs["logits"], batch["labels"])
        elif self._objective_format in ["binary_cross_entropy_with_logits", "volta", "meter"]:
            if "targets" in batch:
                vqa_score_metric = self.get_current_metric("vqa_score_by_labels")
                metric_result = vqa_score_metric(outputs["logits"], batch["targets"])

        if metric_result is not None:
            self.log_dict(
                {result.log_name: result.value for result in metric_result.values()},
                prog_bar=True,
                on_step=True,
                on_epoch=False,
                logger=self.training,
            )
        return outputs

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        outputs = self.step_with_metrics(batch, batch_idx, enhance_outputs=False)
        return outputs

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx=0) -> Dict[str, Any]:
        outputs = self.step_with_metrics(batch, batch_idx, enhance_outputs=True)
        return outputs

    def test_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx=0) -> Dict[str, Any]:
        outputs = self.step_with_metrics(batch, batch_idx, enhance_outputs=True)
        return outputs

    def inference_for_predict(self, batch: Dict[str, Any], **kwargs):
        return self(batch, enhance_outputs=True)
