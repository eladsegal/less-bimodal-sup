from typing import Optional

from src.pl import MyLightningModule
from src.evaluators import HfEvaluator
from src.utils.inspection import get_fqn_hierarchy


class EvaluatorMixin(MyLightningModule):
    def __init__(self, evaluator: Optional[HfEvaluator] = None, **kwargs):
        EvaluatorMixin.add_to_ignore_saving_in_kwargs("evaluator", kwargs)
        super().__init__(**kwargs)
        self._evaluator = evaluator

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        self._run_evaluator()

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()
        self._run_evaluator()

    def _run_evaluator(self):
        if getattr(self, "predictions", None) is not None:
            predictions = self.predictions
            prefix = self.get_current_prefix()
            if self._evaluator is not None:
                # TODO: Supports only HF dataset at the moment
                metrics_to_log = self._evaluator(
                    predictions,
                    self.get_current_dataset_helper(),
                    source_model_hierarchy=get_fqn_hierarchy(type(self)),
                )
                metrics_to_log = {f"{prefix}/{k}": v for k, v in metrics_to_log.items()}
            else:
                metrics_to_log = {f"{prefix}/n_examples": float(len(predictions))}  # Use float to prevent warning

            self.log_dict(
                metrics_to_log,
                prog_bar=False,
                logger=True,
            )
