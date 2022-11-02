import json
from _jsonnet import evaluate_snippet

# Copied from https://github.com/allenai/allennlp/blob/86504e6b57b26bb2bb362e33c0edc3e49c0760fe/allennlp/common/params.py
import copy
import os
from typing import Any, Dict


def evaluate(path):
    with open(path, mode="r") as f:
        jsonnet_str = f.read()
    return json.loads(evaluate_snippet(path, jsonnet_str, ext_vars=_environment_variables()))


def _is_encodable(value: str) -> bool:
    """
    We need to filter out environment variables that can't
    be unicode-encoded to avoid a "surrogates not allowed"
    error in jsonnet.
    """
    # Idiomatically you'd like to not check the != b""
    # but mypy doesn't like that.
    return (value == "") or (value.encode("utf-8", "ignore") != b"")


def _environment_variables() -> Dict[str, str]:
    """
    Wraps `os.environ` to filter out non-encodable values.
    """
    return {key: value for key, value in os.environ.items() if _is_encodable(value)}


def unflatten(flat_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a "flattened" dict with compound keys, e.g.
        {"a.b": 0}
    unflatten it:
        {"a": {"b": 0}}
    """
    unflat: Dict[str, Any] = {}

    for compound_key, value in flat_dict.items():
        curr_dict = unflat
        parts = compound_key.split(".")
        for key in parts[:-1]:
            curr_value = curr_dict.get(key)
            if key not in curr_dict:
                curr_dict[key] = {}
                curr_dict = curr_dict[key]
            elif isinstance(curr_value, dict):
                curr_dict = curr_value
            else:
                raise Exception("flattened dictionary is invalid")
        if not isinstance(curr_dict, dict) or parts[-1] in curr_dict:
            raise Exception("flattened dictionary is invalid")
        curr_dict[parts[-1]] = value

    return unflat


def with_fallback(preferred: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dicts, preferring values from `preferred`.
    """

    def merge(preferred_value: Any, fallback_value: Any) -> Any:
        if isinstance(preferred_value, dict) and isinstance(fallback_value, dict):
            return with_fallback(preferred_value, fallback_value)
        elif isinstance(preferred_value, dict) and isinstance(fallback_value, list):
            # treat preferred_value as a sparse list, where each key is an index to be overridden
            merged_list = fallback_value
            for elem_key, preferred_element in preferred_value.items():
                try:
                    index = int(elem_key)
                    merged_list[index] = merge(preferred_element, fallback_value[index])
                except ValueError:
                    raise Exception(
                        "could not merge dicts - the preferred dict contains "
                        f"invalid keys (key {elem_key} is not a valid list index)"
                    )
                except IndexError:
                    raise Exception(
                        "could not merge dicts - the preferred dict contains "
                        f"invalid keys (key {index} is out of bounds)"
                    )
            return merged_list
        elif isinstance(preferred_value, list) and isinstance(fallback_value, list):
            # eladsegal: merge lists instead of replace, which is more consistent with dictionaries merge
            return copy.deepcopy(fallback_value) + copy.deepcopy(preferred_value)
        else:
            return copy.deepcopy(preferred_value)

    preferred_keys = set(preferred.keys())
    fallback_keys = set(fallback.keys())
    common_keys = preferred_keys & fallback_keys

    merged: Dict[str, Any] = {}

    for key in preferred_keys - fallback_keys:
        merged[key] = copy.deepcopy(preferred[key])
    for key in fallback_keys - preferred_keys:
        merged[key] = copy.deepcopy(fallback[key])

    for key in common_keys:
        preferred_value = preferred[key]
        fallback_value = fallback[key]

        merged[key] = merge(preferred_value, fallback_value)
    return merged


def parse_jsonnet_overrides(serialized_overrides: str) -> Dict[str, Any]:
    if serialized_overrides:
        ext_vars = _environment_variables()

        return unflatten(json.loads(evaluate_snippet("", serialized_overrides, ext_vars=ext_vars)))
    else:
        return {}
