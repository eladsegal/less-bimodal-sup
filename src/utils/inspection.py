import sys
import inspect


def find_in_module(obj, name_to_find):
    for base_klass in get_class_hierarchy(obj):
        module = sys.modules[base_klass.__module__]
        if name_to_find in dir(module):
            return getattr(module, name_to_find)
    return None


def find_in_class(klass, name_to_find, ignore_klass):
    for base in get_class_hierarchy(klass):
        if base == ignore_klass or issubclass(ignore_klass, base):
            continue
        result = getattr(base, name_to_find, None)
        if result is not None and result != getattr(ignore_klass, name_to_find):
            return result
    return None


def get_class_hierarchy(klass):
    return inspect.getmro(klass)


def get_fqn(klass):
    return klass.__module__ + "." + klass.__qualname__


def get_fqn_hierarchy(klass):
    return [get_fqn(c) for c in get_class_hierarchy(klass)]
