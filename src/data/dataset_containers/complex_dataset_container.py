from typing import Dict, Any


class ComplexDatasetContainer:
    def __init__(
        self,
        main_key: str,
        additional_kwargs: Dict[str, Any] = None,
        **kwargs,
    ):
        self._internal_dict = dict(**kwargs)
        self.main_key = main_key
        self.additional_kwargs = additional_kwargs if additional_kwargs is not None else {}

    def __getitem__(self, key):
        return self._internal_dict[key]

    def __iter__(self):
        return iter(self._internal_dict)

    def __len__(self):
        return len(self._internal_dict)

    def __contains__(self, key):
        return key in self._internal_dict

    def keys(self):
        return self._internal_dict.keys()

    def items(self):
        return self._internal_dict.items()

    def values(self):
        return self._internal_dict.values()

    def setup(self, additional_kwargs: Dict[str, Any] = None, kwargs_per_container: Dict[str, Any] = None, **kwargs):
        if additional_kwargs is None:
            additional_kwargs = self.additional_kwargs

        self.seed = additional_kwargs["seed"] if "seed" in additional_kwargs else 0

        kwargs_per_container = kwargs_per_container if kwargs_per_container is not None else {}
        for key, dataset_container in self.items():
            kwargs.update(kwargs_per_container.get(key, {}))
            dataset_container.setup(self, additional_kwargs=additional_kwargs.get(key, additional_kwargs), **kwargs)
