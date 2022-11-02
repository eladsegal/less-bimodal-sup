from typing import Generic, TypeVar, Type, Any
from hydra._internal.utils import _locate

T = TypeVar("T")


class Instantiation(Generic[T]):
    def __init__(self, klass: Type[T], **kwargs) -> None:
        self._klass = klass
        self.kwargs = kwargs

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        return self._klass(*args, **self.kwargs, **kwargs)


def get_class_type(full_class_name):
    return _locate(full_class_name)
