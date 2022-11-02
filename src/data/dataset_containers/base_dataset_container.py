from typing import Optional, Dict, Any

from src.data.dataset_containers.complex_dataset_container import ComplexDatasetContainer


class BaseDatasetContainer:
    def __init__(
        self,
        additional_kwargs: Dict[str, Any] = None,
    ):
        self._has_setup_run = False
        self.raw_datasets = None  #  Should be a mapping from a split to a dataset
        self.ready_datasets = None  #  Should be a mapping from a split to a dataset

        self.additional_kwargs = additional_kwargs if additional_kwargs is not None else {}

    def setup(
        self,
        parent_container: Optional[ComplexDatasetContainer] = None,
        additional_kwargs: Dict[str, Any] = None,
        **kwargs
    ):
        # TODO: Should not pass the parrent container, just the values that are needed
        if self._has_setup_run:
            return
        else:
            self._has_setup_run = True

        if additional_kwargs is None:
            additional_kwargs = {}

        if "seed" in additional_kwargs:
            self.seed = additional_kwargs["seed"]
            self._using_default_seed = False
        else:
            self.seed = 0
            self._using_default_seed = True

        self._setup(parent_container=parent_container, additional_kwargs=additional_kwargs, **kwargs)

    def _setup(self):
        raise NotImplementedError
