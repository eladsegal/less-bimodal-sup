import collections
import os
import logging
import shlex
import subprocess

from contextlib import contextmanager

logger = logging.getLogger(__name__)

from hydra.utils import get_original_cwd
from hydra.core.global_hydra import GlobalHydra


def get_source_code_path():
    if GlobalHydra.instance().is_initialized():
        return get_original_cwd()
    else:
        return os.getcwd()


def resolve_relative_paths(paths):
    temp_cwd = os.getcwd()
    if GlobalHydra.instance().is_initialized():
        os.chdir(get_original_cwd())
    paths = _resolve_relative_paths_helper(paths)
    os.chdir(temp_cwd)
    return paths


def _resolve_relative_paths_helper(paths):
    if paths is None:
        return None
    elif isinstance(paths, str):
        new_paths = os.path.abspath(paths)
    elif isinstance(paths, collections.abc.Mapping):
        new_paths = {}
        for k, v in paths.items():
            new_paths[k] = _resolve_relative_paths_helper(v)
    elif isinstance(paths, collections.abc.Sequence):
        new_paths = []
        for path in paths:
            new_paths.append(_resolve_relative_paths_helper(path))
    else:
        raise ValueError("Illegal `paths` type")
    return new_paths


@contextmanager
def smart_open(*args, **kwargs):
    if len(args) > 0:
        file = args[0]
    else:
        file = kwargs["file"]
    if len(args) > 1:
        mode = args[1]
    else:
        mode = kwargs["mode"]
    if any(write_mode in mode for write_mode in ["w", "x", "a", "+"]):
        if not os.path.isdir(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(*args, **kwargs) as f:
        yield f


# Taken from https://stackoverflow.com/questions/568271/how-to-check-if-there-exists-a-process-with-a-given-pid-in-python
def pid_exists(pid):
    """Check For the existence of a unix pid."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def fast_listdir(path, target_type):
    fast_listdir_path = os.path.join(get_source_code_path(), "tools/fast_listdir")
    if not os.path.isfile(f"{fast_listdir_path}"):
        subprocess.run(f"gcc {fast_listdir_path}.c -o {fast_listdir_path}".split(), check=True, capture_output=True)
    process = subprocess.run(f"{fast_listdir_path} {shlex.quote(path)} {target_type}".split(), capture_output=True)
    p_out = process.stdout.strip().decode("utf-8")
    p_err = process.stderr.strip().decode("utf-8")
    if process.returncode != 0:
        raise Exception(p_err)
    result = p_out.split("\n")
    if len(result) > 0 and result[-1] == "":
        result = result[:-1]
    return result


# TODO: P0 this should be done once per directory
def get_files_with_extensions(path, extensions, recursive=True, secondary_call=False, fast=True, progress=None):
    if not secondary_call:
        from src.pl.callbacks.progress import tqdm

        progress = tqdm(desc=f"Getting files with extensions {extensions} in {path}")

    if isinstance(extensions, str):
        extensions = [extensions]

    if not secondary_call:
        logger.info(f"Searching in {path}")
    result = []
    if fast:
        if recursive:
            for name in fast_listdir(path, "d"):
                result.extend(
                    get_files_with_extensions(
                        os.path.join(path, name),
                        extensions,
                        recursive,
                        secondary_call=True,
                        fast=fast,
                        progress=progress,
                    )
                )

        for name in fast_listdir(path, "f"):
            handle_file(os.path.join(path, name), result, extensions, progress)

        for name in fast_listdir(path, "u"):
            # checking isdir is too slow, so this is a hacky workaround
            child_path = os.path.join(path, name)
            length_before = len(result)
            handle_file(child_path, result, extensions, progress)
            length_after = len(result)
            if length_before == length_after:
                if os.path.isdir(child_path):
                    result.extend(
                        get_files_with_extensions(
                            child_path, extensions, recursive, secondary_call=True, fast=fast, progress=progress
                        )
                    )
    else:
        for name in os.listdir(path):
            assert len(name) != 0, path
            child_path = os.path.join(path, name)
            if os.path.isdir(child_path) and recursive:
                result.extend(
                    get_files_with_extensions(
                        child_path, extensions, recursive, secondary_call=True, fast=fast, progress=progress
                    )
                )
            elif os.path.isfile(child_path):
                handle_file(child_path, result, extensions, progress)
    if not secondary_call:
        progress.close()
    return result


def handle_file(path, result, extensions, progress):
    if any(path.endswith(extension) for extension in extensions):
        progress.update()
        result.append(path)
