from pytorch_lightning.trainer.states import RunningStage

STAGE_TO_SPLIT = {
    RunningStage.SANITY_CHECKING: "validation",
    RunningStage.TRAINING: "train",
    RunningStage.VALIDATING: "validation",
    RunningStage.TESTING: "test",
    RunningStage.PREDICTING: "test",
    RunningStage.TUNING: "train",
}

from src.data.dataset_helper import DatasetHelper
