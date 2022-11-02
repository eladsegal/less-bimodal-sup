from collections.abc import Mapping, Sequence
import os

from utils.dot_notation import get_dot, set_dot
from src.utils.config import safe_open_dict


def resolve_path_priorities(path_priorities):
    existing_paths = None
    if isinstance(path_priorities, Sequence) and not isinstance(path_priorities, str):
        # multiple options
        if len(path_priorities) > 0:
            first_elem = path_priorities[0]
            if isinstance(first_elem, Sequence) and not isinstance(first_elem, str):
                # path_priorities is List[List[str]], need to choose List[str]
                # where all str are paths
                for paths in path_priorities:
                    if all(os.path.exists(path) for path in paths):
                        existing_paths = paths
                        break
            elif isinstance(first_elem, Mapping):
                # path_priorities is List[Dict[str, str]], need to choose Dict[str, str]
                # where all values are paths
                for paths_dict in path_priorities:
                    if all(os.path.exists(path) for path in paths_dict.values()):
                        existing_paths = paths_dict
                        break
            elif isinstance(first_elem, str):
                # path priorities is List[str], need to choose str
                # that is a path
                for path in path_priorities:
                    if os.path.exists(path):
                        existing_paths = path
                        break
    elif isinstance(path_priorities, Mapping):
        # single option
        paths_dict = path_priorities
        if all(os.path.exists(path) for path in paths_dict.values()):
            existing_paths = paths_dict
    elif isinstance(path_priorities, str):
        # single option
        path = path_priorities
        if os.path.exists(path):
            existing_paths = path

    return existing_paths


def config_modifications_for_path_priorities(cfg):
    """
    Support the case where the required files are not in the same directory in each of the machines that are used.
    I'd happy to get rid of this, probably the best way would be with symlinks in the repo folder or above in a path that is not synced.
    """
    for path_priorities_key in cfg["global"].get("path_priorities_keys", []):
        exists, cfg_value = get_dot(cfg, path_priorities_key)
        if not exists:
            continue

        if isinstance(cfg_value, Sequence):
            existing_paths = resolve_path_priorities(cfg_value)

            if existing_paths is None:
                raise Exception(f"Couldn't find existing paths for {path_priorities_key}")

            with safe_open_dict(cfg):
                set_dot(cfg, path_priorities_key, existing_paths)
        elif isinstance(cfg_value, Mapping):
            for key, path_priorities in cfg_value.items():
                existing_paths = resolve_path_priorities(path_priorities)

                if existing_paths is None:
                    raise Exception(f"Couldn't find existing paths in {path_priorities_key} for {key}")

                with safe_open_dict(cfg):
                    set_dot(cfg, f"{path_priorities_key}.{key}", existing_paths)

    with safe_open_dict(cfg):
        cfg["global"].pop("path_priorities_keys", None)
