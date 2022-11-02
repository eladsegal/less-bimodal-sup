from copy import deepcopy

from transformers import PreTrainedTokenizerBase

from src.utils.inspection import find_in_class


class Mapper:
    def get_mapping_fn(self, split):
        if split == "test":
            return self.test_mapping
        elif split == "validation":
            return self.val_mapping
        elif split == "train":
            return self.train_mapping
        return self.general_mapping

    def get_mapping_kwargs(self, split, dataset, **kwargs):
        if split == "test":
            mapping_kwargs = self.test_mapping_kwargs
        elif split == "validation":
            mapping_kwargs = self.val_mapping_kwargs
        elif split == "train":
            mapping_kwargs = self.train_mapping_kwargs
        else:
            mapping_kwargs = self.general_mapping_kwargs

        # The tokenizer changes after it's used with some arguments (e.g. max_length, truncation),
        # therefore we make a copy in order to have the same fingerprint when using .map.
        # This is cheaper than mapping when we can already load from the cache.
        for key, value in mapping_kwargs.items():
            if isinstance(value, PreTrainedTokenizerBase):
                mapping_kwargs[key] = deepcopy(value)

        mapping_kwargs["split"] = split
        if "dataset" in mapping_kwargs:
            mapping_kwargs["dataset"] = dataset

        mapping_kwargs.update(kwargs)
        return mapping_kwargs

    @property
    def general_mapping_kwargs(self):
        raise NotImplementedError

    @property
    def general_mapping(self):
        func = find_in_class(type(self), "general_mapping", ignore_klass=Mapper)
        if func is not None:
            return func
        else:
            raise NotImplementedError

    # If not overriden, fall back
    @property
    def train_mapping_kwargs(self):
        return self.general_mapping_kwargs

    @property
    def train_mapping(self):
        func = find_in_class(type(self), "train_mapping", ignore_klass=Mapper)
        return func if func is not None else self.general_mapping

    @property
    def val_mapping_kwargs(self):
        return self.train_mapping_kwargs

    @property
    def val_mapping(self):
        func = find_in_class(type(self), "val_mapping", ignore_klass=Mapper)
        return func if func is not None else self.train_mapping

    @property
    def test_mapping_kwargs(self):
        return self.val_mapping_kwargs

    @property
    def test_mapping(self):
        func = find_in_class(type(self), "test_mapping", ignore_klass=Mapper)
        return func if func is not None else self.val_mapping
