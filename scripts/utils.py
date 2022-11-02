from typing import Optional, Union, List
from collections.abc import Mapping, Sequence
from collections import defaultdict

import os
import sys
import argparse
import shlex
import subprocess
from pathlib import Path
from functools import cache
import subprocess

from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.override_parser.types import OverrideType, ValueType

from scripts.namegenerator import gen as namegenerator
from utils.redis_lock import redis_lock

PARENT_OUTPUT_DIR = "../outputs"
FILE_NAME_MAX_LENGTH = 215


def safe_execute(default, function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except:
        return default


@cache
def has_slurm():
    return subprocess.run("sinfo", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True).returncode == 0


def is_on_slurm():
    return "SLURM_JOB_NAME" in os.environ


def get_mapping(keyword, locals):
    mapping = defaultdict(lambda: None)
    for key, value in locals.items():
        if key.startswith(keyword):
            mapping[key.replace(keyword, "", 1)] = value
    return mapping


def prep_command(command_parts: Union[str, List[str]], slurm_config, unknown, execute_args):
    assert command_parts is not None
    if isinstance(command_parts, str):
        command_parts = [command_parts]

    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["OC_CAUSE"] = "1"
    os.environ["DEBUG"] = os.environ.get("DEBUG", "false")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    command = get_command_from_command_parts(command_parts + list(map(shlex.quote, unknown)))

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output_dir", type=str, default=None)
    args, _ = parser.parse_known_args(get_sys_argv(command)[1:])

    output_dir = None
    if not execute_args.multirun:
        if args.output_dir is None:
            config = get_config_from_command(command)

            if config is not None:
                output_dir = config["metadata"].get("output_dir")

                if slurm_config is not None and "gpus-per-node" not in slurm_config:
                    slurm_config["gpus-per-node"] = config["trainer"].get("gpus", 0)
                if slurm_config is not None and "nodes" not in slurm_config:
                    slurm_config["nodes"] = config["trainer"].get("num_nodes", 1)

            if output_dir is not None:
                command = " ".join([command, "--output_dir", output_dir])
            else:
                output_dir, unique_name = get_output_dir(PARENT_OUTPUT_DIR, get_sys_argv(command))
        else:
            output_dir = args.output_dir

    return {
        "command": command,
        "output_dir": output_dir,
        "slurm_config": slurm_config,
    }


def get_config_from_command(command):
    sys_argv = get_sys_argv(command)
    if any(
        entry_point in command
        for entry_point in [
            "src/train.py",
            "src/validate.py",
            "src/test.py",
            "src/find_batch_size.py",
            "src/iterate_over_data.py",
        ]
    ):
        try:
            from src.utils.config import EXTENSION_TO_CONFIG_HANDLER
        except Exception:
            EXTENSION_TO_CONFIG_HANDLER = {}

        if not (
            len(sys_argv) > 1
            and (
                os.path.isdir(sys_argv[1])
                or any(
                    sys_argv[1].endswith(extension)
                    for extension in EXTENSION_TO_CONFIG_HANDLER
                    if extension is not None
                )
            )
        ):
            print(sys_argv)
            raise Exception("The first argument should be a directory or a file with a config extension")

        from src.utils.config import handle_config

        original_argv = list(sys_argv)
        sys.argv = sys_argv  # required for handle_config, should be just like the executed command
        config, _ = handle_config(request_input=False, from_execute=True)
        sys.argv = original_argv
        return config
    return None


def get_command_from_command_parts(command_parts):
    return " ".join(
        map(
            lambda command_part: command_part.strip()[:-1].strip()
            if command_part.strip().endswith("\\")
            else command_part.strip(),
            command_parts,
        )
    )


def get_sys_argv(command):
    command_split = shlex.split(command)

    # Prepare the command line arguments
    if command_split[0] == "python":
        if command_split[1] == "-m":
            start_index = 2
        else:
            start_index = 1
    else:
        start_index = 0

    if start_index > 0 and not command_split[start_index].endswith(".py"):
        command_split[start_index] = command_split[start_index].replace(".", "/") + ".py"

    return command_split[start_index:]


DELETE_VALUE = "_delete_"
NULL_VALUE = "_null_"


def handle_args_to_delete_or_null(config):
    if isinstance(config, Mapping):
        keys_to_delete = []
        keys_to_null = []
        for key, value in config.items():
            if value == DELETE_VALUE:
                keys_to_delete.append(key)
            elif value == NULL_VALUE:
                keys_to_null.append(key)
            else:
                handle_args_to_delete_or_null(value)
        for key in reversed(keys_to_delete):
            del config[key]
        for key in reversed(keys_to_null):
            config[key] = None
    elif isinstance(config, Sequence) and not isinstance(config, str):
        indices_to_delete = []
        indices_to_null = []
        for i, value in enumerate(config):
            if value == DELETE_VALUE:
                indices_to_delete.append(i)
            elif value == NULL_VALUE:
                indices_to_null.append(i)
            else:
                handle_args_to_delete_or_null(value)
        for i in reversed(indices_to_delete):
            del config[i]
        for i in reversed(indices_to_null):
            config[i] = None


def get_parsed_overrides(overrides_list):
    overrides_list = list(overrides_list)
    special_chars = "="  # "\()[]{}:=,"
    for i in range(len(overrides_list)):
        if any(char in overrides_list[i] for char in ['"', "'", " ", "\t"]):
            continue
        for char in special_chars:
            overrides_list[i] = overrides_list[i].replace(char, f"\{char}")
            if char == "=":
                overrides_list[i] = overrides_list[i].replace("\=", "=", 1)

    parser = OverridesParser.create()
    parsed_overrides = parser.parse_overrides(overrides=overrides_list)
    for parsed_override in parsed_overrides:
        if parsed_override.type in [OverrideType.CHANGE, OverrideType.ADD]:
            parsed_override.type = OverrideType.FORCE_ADD
    return parsed_overrides


def get_output_dir(parent_output_dir: str, sys_argv, config_name: Optional[str] = None):
    # from datetime import datetime
    # time_str = datetime.now().strftime("%Y-%m-%d") + "_" + datetime.now().strftime("%H-%M-%S")
    generated_name = namegenerator(n=2)
    numbered_generated_name = f"{get_number(PARENT_OUTPUT_DIR)}_{generated_name}"

    name_components = [numbered_generated_name]

    if sys_argv[0] not in ["src/train.py"]:
        output_name_prefix = Path(sys_argv[0]).stem
        name_components.append(output_name_prefix)

    if config_name is not None:
        name_components.append(config_name)

    full_name = "-".join(name_components)
    output_dir = os.path.realpath(os.path.join(parent_output_dir, full_name))
    output_dir = shorten_file_name(output_dir, FILE_NAME_MAX_LENGTH)

    return output_dir, numbered_generated_name


def shorten_file_name(file_path, max_length):
    new_file_path = file_path[:max_length]
    if os.path.dirname(new_file_path) != os.path.dirname(file_path):
        raise Exception(f"The file name {file_path} is too long")
    return new_file_path


def get_number(number_dir):
    os.makedirs(number_dir, exist_ok=True)
    file_path = os.path.abspath(os.path.join(f"{number_dir}", "number.txt"))
    with redis_lock(file_path):
        if not os.path.isfile(file_path):
            with open(file_path, "w") as f:
                f.write("0")
        with open(file_path, mode="r") as f:
            number = int(f.read())
        new_number = number + 1
        with open(file_path, mode="w") as f:
            f.write(str(new_number))
    return new_number


def override_to_value(override):
    if override.value_type == ValueType.ELEMENT:
        return override.value()
    elif override.value_type == ValueType.SIMPLE_CHOICE_SWEEP:
        return override.value().list
    elif override.value_type == ValueType.RANGE_SWEEP:
        range_args = override.value()
        return list(range(range_args.start, range_args.stop, range_args.step))
