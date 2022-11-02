from typing import Dict, Iterable, List, Union, Optional, Any
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import argparse
from argparse import Namespace
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
import shutil
from _jsonnet import evaluate_snippet
from hydra.core.override_parser.types import ValueType
import yaml
import psutil
import shlex
import contextlib

from omegaconf import OmegaConf, DictConfig, open_dict
from hydra._internal.config_loader_impl import ConfigLoaderImpl
from hydra.core.override_parser.types import Override

from scripts.utils import (
    get_parsed_overrides,
    handle_args_to_delete_or_null,
    get_output_dir,
    PARENT_OUTPUT_DIR,
    FILE_NAME_MAX_LENGTH,
    shorten_file_name,
)

from utils.jsonnet import with_fallback, _environment_variables
from utils.dot_notation import set_dot

TEMPLATES_DIR = "configs/templates"
METADATA_PATH = "metadata"
LEGAL_KEYS = [
    "dataset_container",
    "metrics",
    "datamodule",
    "trainer",
    "model",
    "global",
    "metadata",
    "defaults",
    "hydra",
]
HIDDEN_KEYS = ["callbacks_dict"]


def hydra_prep_with_multiprocessing_support(random_temp_name, **kwargs):
    original_argv = list(sys.argv)

    if is_config_yaml(sys.argv[1]):
        shutil.copyfile(sys.argv[1], os.path.join(TEMPLATES_DIR, f"{random_temp_name}.yaml"))
        sys.argv = [sys.argv[0]] + sys.argv[2:]
    else:
        _, yaml_path = handle_config(random_temp_name, **kwargs)
        original_argv = list(sys.argv)
        original_argv.insert(1, yaml_path)

    return original_argv


def handle_config(
    random_temp_name: Optional[str] = None,
    *,
    is_eval: bool = False,
    mandatory_verification: bool = True,
    is_silent: bool = False,
    request_input: Optional[bool] = None,
    restore_argv=False,
    sys_argv=None,
    from_execute: bool = False,
):
    """
    Consumes command line arguments of the config argument parser, and also consumes the overrides with hydra
    """
    # TODO: Delegate all arguments handling to hydra.core.override_parser.parse_overrides, stop using sys.argv
    using_actual_sys_argv = False
    if sys_argv is None:
        using_actual_sys_argv = True
        sys_argv = sys.argv

    if request_input is None:
        if sys_argv[0] in ["src/train.py", "src/validate.py", "src/test.py"]:
            request_input = False
        else:
            request_input = False

    if sys_argv[0] in ["src/validate.py", "src/test.py"]:
        is_eval = True

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("path", type=str, help=".jsonnet/.ckpt path")
    parser.add_argument("--u", dest="unify_files", action="append", default=[])
    parser.add_argument("--m", dest="merge_files", action="append", default=[])
    parser.add_argument("-n", "--notes", type=str, default=None)
    parser.add_argument("--queue_id", type=str, default=None)
    parser.add_argument("--queue_eid", type=str, default=None)
    parser.add_argument("--request_input", action="store_true", default=False)
    parser.add_argument("--config_name_ignore_keys", type=str, default="")
    parser.add_argument(
        "--create_new_output_dir",
        action="store_true",
        default=False,
        help="When resuming, create a new output dir instead of using the existing one.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    args, unknown = parser.parse_known_args(sys_argv[1:])

    if args.output_dir is not None and not from_execute:
        # Existence of a preemption file only make use of the overrides
        preemption_file = os.path.join(args.output_dir, "preempted")
        if os.path.isfile(preemption_file):
            args.path = args.output_dir
            args.unify_files = []
            args.merge_files = []
            args.notes = None
            args.queue_id = None
            args.queue_eid = None
            args.request_input = False
            args.config_name_ignore_keys = ""
            args.create_new_output_dir = False
            args.output_dir = None

    original_argv = list(sys_argv)
    original_command = " ".join(["python"] + list(map(shlex.quote, sys_argv)))

    sys_argv = [sys_argv[0]]

    unify_and_merge_as_overrides = get_parsed_overrides(
        [f"u={arg}" for arg in args.unify_files] + [f"m={arg}" for arg in args.merge_files]
    )
    check_for_sweeps(unify_and_merge_as_overrides)

    original_parsed_overrides = [
        parsed_override
        for parsed_override in get_parsed_overrides(
            [arg for arg in unknown if not (arg.startswith("slurm.") and "=" in arg)]
        )
    ]
    check_for_sweeps(original_parsed_overrides)

    parsed_jsonnet_overrides = [
        parsed_override
        for parsed_override in original_parsed_overrides
        if not parsed_override.key_or_group.startswith("o__")
    ]

    parsed_overrides = []
    for parsed_override in original_parsed_overrides:
        if not parsed_override.key_or_group.startswith("o__"):
            continue

        # if not any(parsed_override.key_or_group.startswith(f"{legal_key}.") for legal_key in LEGAL_KEYS):
        #     parsed_override.key_or_group = f"global.{parsed_override.key_or_group}"
        parsed_overrides.append(parsed_override)

    if os.path.isfile(args.path):
        extension = Path(args.path).suffix
    elif os.path.isdir(args.path):
        extension = None
    else:
        raise ValueError(f"{args.path} is not a file or directory")

    config_handler = EXTENSION_TO_CONFIG_HANDLER[extension](
        sys_argv=sys_argv,
        args=args,
        parsed_jsonnet_overrides=parsed_jsonnet_overrides,
        parsed_overrides=parsed_overrides,
        random_temp_name=random_temp_name,
        is_eval=is_eval,
        mandatory_verification=mandatory_verification,
        is_silent=is_silent,
        original_command=original_command,
    )
    if request_input is False or isinstance(config_handler, CkptConfigHandler):
        args.request_input = False

    config, yaml_path = config_handler()

    if restore_argv:
        sys_argv = original_argv

    if using_actual_sys_argv:
        sys.argv = sys_argv

    return config, yaml_path


def check_for_sweeps(parsed_overrides):
    if not all(parsed_override.value_type == ValueType.ELEMENT for parsed_override in parsed_overrides):
        raise Exception("To do command line sweeps, pass the --multirun flag")


def is_config_yaml(value):
    if not os.path.isfile(value):
        return False
    file_name = os.path.basename(value)
    return file_name.endswith(".yaml") and os.path.realpath(TEMPLATES_DIR) == os.path.commonpath(
        [os.path.realpath(TEMPLATES_DIR), os.path.realpath(value)]
    )


@dataclass
class ConfigHandler:
    sys_argv: List[str]
    args: Namespace
    parsed_jsonnet_overrides: List[Override]
    parsed_overrides: List[Override]
    original_command: Optional[str]
    random_temp_name: Optional[str] = None
    is_eval: bool = False
    is_silent: bool = False
    resume: bool = False
    mandatory_verification: bool = True

    def __call__(self):
        config = self._obtain_config()
        config = unify(config, self.args.unify_files, self.parsed_jsonnet_overrides)
        config = merge(config, self.args.merge_files)
        handle_args_to_delete_or_null(config)
        if "metadata" not in config:
            config["metadata"] = {}

        # TODO: Move all the code with the assumptions to a new class, PyTorchLightningConfigHandler.
        # ConfigHandler class should be general as possible.

        config["defaults"] = (["hydra"] if not self.is_silent else ["hydra_silent"]) + ["_self_"]
        config["hydra"] = {}

        self._edit_config(config)

        config["metadata"]["job_type"] = Path(self.sys_argv[0]).stem

        wandb_logger = get_wandb_logger_config(config)
        if wandb_logger is not None:
            wandb_logger["job_type"] = config["metadata"]["job_type"]

        if not self.resume or self.args.create_new_output_dir or self.is_eval:
            if not self.is_silent:
                output_dir, unique_name = self._get_output_dir(config)

                if wandb_logger is not None:
                    wandb_logger.pop("id", None)
                    wandb_logger["name"] = os.path.basename(output_dir)

                config["metadata"]["output_dir"] = os.path.realpath(output_dir)
                if unique_name is not None:
                    config["metadata"]["unique_name"] = unique_name

            config_name = self._get_config_name(config)
            assert config_name is not None
            config["metadata"]["config_name"] = config_name

        if not self.is_silent:
            config["hydra"]["run"] = {"dir": config["metadata"]["output_dir"]}
        config["hydra"]["job"] = {"name": config["metadata"]["config_name"]}

        notes = self._get_notes(config, self.args.notes, self.args.request_input)
        if notes is not None:
            if wandb_logger is not None:
                wandb_logger["notes"] = notes
            else:
                config["notes"] = notes

        config["metadata"]["command"] = self.original_command
        if wandb_logger is not None:
            wandb_logger["command"] = self.original_command

        config = OmegaConf.create(config)
        parsed_overrides = deepcopy(self.parsed_overrides)
        for parsed_override in parsed_overrides:
            parsed_override.key_or_group = parsed_override.key_or_group.replace("o__", "", 1)
        ConfigLoaderImpl._apply_overrides_to_config(parsed_overrides, config)
        config = OmegaConf.to_container(config)
        handle_args_to_delete_or_null(config)

        self._temp_edit_config(config)

        if self.mandatory_verification:
            verify_mandatory_fields(config)

        # if number of gpus is not specified, set to 1
        # if the specified value is bigger than 1, use the ddp plugin
        if "trainer" in config:
            gpus = config["trainer"].get("gpus", 0)
            if gpus > 1:
                config = merge(config, ["configs/pieces/snippets/ddp.jsonnet"], override=False)
            elif gpus <= 1:
                config["trainer"].pop("strategy", None)

        keys_diff = sorted(list(set(config.keys()) - set(LEGAL_KEYS)))
        assert (
            len(keys_diff) == 0
        ), f"Only {[legal_key for legal_key in LEGAL_KEYS if legal_key not in ['metadata', 'defaults', 'hydra']]} are allowed in the config, but found {keys_diff}"

        if self.random_temp_name is not None:
            # used in @hydra.main
            yaml_path = os.path.join(TEMPLATES_DIR, f"{self.random_temp_name}.yaml")
            with open(yaml_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            return config, yaml_path
        else:
            return config, None

    def _edit_config(self, config):
        if self.sys_argv[0] in ["src/iterate_over_data.py"]:
            config["trainer"]["gpus"] = 0
            config["trainer"]["num_nodes"] = 1
            config["trainer"].pop("strategy", None)
        elif self.sys_argv[0] in ["src/find_batch_size.py"]:
            config["trainer"]["gpus"] = 1
            config["trainer"]["num_nodes"] = 1
            config["trainer"].pop("strategy", None)

        # TODO: Should be defined in a modular way, outside
        # try:
        if config["global"]["dataset_name"] in ["vqa", "gqa", "nlvr2"]:
            load_weights = config["global"].get("load_weights")
            if load_weights is not None and load_weights != "DUMMY_PATH":
                import torch

                model_dict = torch.load(load_weights, map_location="cpu")
                load_weights_config = model_dict["hyper_parameters"]["cfg"]
                config["global"]["extras"] = config["global"].get("extras", {})
                if load_weights_config["global"]["dataset_name"] in [
                    "conceptual_captions",
                    "conceptual_captions_labels",
                ]:
                    cc_steps = model_dict["global_step"]
                    cc_percentage = load_weights_config["dataset_container"]["language"].get("reduced_train_size")

                    config["global"]["extras"]["cc_steps"] = cc_steps
                    config["global"]["extras"]["cc_percentage"] = cc_percentage if cc_percentage is not None else 1.0
                    config["global"]["extras"]["label_corruption_rate"] = config["datamodule"].get(
                        "label_corruption_rate"
                    )
                    config["global"]["extras"]["confidence_threshold"] = (
                        config["dataset_container"]["language"].get("mapper_kwargs", {}).get("confidence_threshold")
                    )
                    config["global"]["extras"]["max_num_of_labels"] = (
                        config["dataset_container"]["language"].get("mapper_kwargs", {}).get("max_num_of_labels")
                    )
                elif load_weights_config["global"]["dataset_name"] in ["vqa", "gqa"]:
                    cc_percentage = load_weights_config["dataset_container"]["language"].get("reduced_train_size")
                    config["global"]["extras"]["pretraining_data_percentage"] = (
                        cc_percentage if cc_percentage is not None else 1.0
                    )
        # except:

    def _temp_edit_config(self, config):
        # TODO: Find a more suitable place for these, or pass with o__ prefix
        if (
            "model" in config
            and (
                "clip" in config["model"].get("pretrained_vision_model", "")
                or "microsoft/beit-base-patch16-224-pt22k" == config["model"].get("pretrained_vision_model", "")
            )
            and "trainer" in config
            and config["trainer"].get("gpus", 0) > 1
        ):
            if "strategy" not in config:
                config["trainer"]["strategy"] = {}
            config["trainer"]["strategy"]["find_unused_parameters"] = True

        if self.args.queue_id is not None:
            queue_id = self.args.queue_id
            config["metadata"]["queue_id"] = queue_id
            if "_" in queue_id:
                config["metadata"]["queue_name"] = queue_id[queue_id.index("_") + 1 : queue_id.rindex("_")]
        if self.args.queue_eid is not None:
            config["metadata"]["queue_eid"] = self.args.queue_eid

        if config["global"].get("debug", False):
            config["datamodule"]["dataloader_num_workers"] = 0
            config["datamodule"]["map_num_proc"] = 1
            config["dataset_container"]["additional_kwargs"]["language"]["map_num_proc"] = 1

    def _get_config_name(self, config):
        used_configs = []
        used_configs.extend(self.args.unify_files)
        used_configs.extend(self.args.merge_files)

        parts = []
        if len(used_configs) > 0:
            parts.append("-".join(map(lambda path: Path(path).stem, used_configs)))

        with open("settings/config_name_ignore_keys.txt", mode="r") as f:
            config_name_ignore_keys = set(f.read().splitlines())
        config_name_ignore_keys.update(self.args.config_name_ignore_keys.split(","))
        config_name_ignore_keys = set(key.strip() for key in config_name_ignore_keys)

        with open("settings/config_name_key_mappings.json", mode="r") as f:
            config_name_key_mappings = json.load(f)
        with open("settings/config_name_value_mappings.json", mode="r") as f:
            config_name_value_mappings = json.load(f)

        for override_type in ["parsed_jsonnet_overrides", "parsed_overrides"]:
            parsed_overrides = getattr(self, override_type)
            if len(parsed_overrides) > 0:
                unknown_dedups = {}
                for parsed_override in parsed_overrides:
                    arg_key = parsed_override.key_or_group.replace("o__", "", 1)
                    if arg_key in config_name_ignore_keys:
                        continue
                    arg_key = config_name_key_mappings.get(arg_key, arg_key)

                    arg_value = str(parsed_override.value())
                    arg_value = config_name_value_mappings.get(arg_value, arg_value)

                    # Handle long unknowns when using checkpoints
                    if arg_value.endswith(".ckpt"):
                        if os.path.dirname(arg_value).endswith("checkpoints"):
                            ckpt_identifier = Path(arg_value).stem
                            if "_" in ckpt_identifier:
                                ckpt_identifier = ckpt_identifier[ckpt_identifier.rindex("_") + 1 :]

                            other_output_dir = os.path.basename(os.path.dirname(os.path.dirname(arg_value)))
                            unique_id = (
                                other_output_dir[: other_output_dir.index("-")]
                                if "-" in other_output_dir
                                else other_output_dir
                            )
                            arg_value = f"{unique_id}_{ckpt_identifier}"
                        else:
                            arg_value = os.path.basename(arg_value)
                    arg_value = arg_value.replace("/", "_")

                    unknown_dedups[arg_key] = arg_value

                if len(unknown_dedups) > 0:
                    parts.append("-".join((f"{key}={value}" for key, value in sorted(unknown_dedups.items()))))

        return "-".join(parts)

    def _get_notes(self, config, notes, request_input):
        if notes is None:
            if request_input:
                from scripts.utils import is_on_slurm  # to prevent circular import

                on_sweep = "WANDB_SWEEP_ID" in os.environ
                if not is_on_slurm() and not on_sweep:
                    from pytorch_lightning.utilities import rank_zero_only

                    if rank_zero_only.rank == 0:
                        notes = input("Input experiment notes:\n").strip() if notes is None else notes
                        if len(notes) == 0:
                            notes = None
        return notes

    def _get_output_dir(self, config):
        unique_name = None
        if self.args.output_dir is None:
            parent_output_dir = (
                os.path.join(PARENT_OUTPUT_DIR, self.args.queue_id)
                if self.args.queue_id is not None
                else PARENT_OUTPUT_DIR
            )
            output_dir, unique_name = get_output_dir(parent_output_dir, self.sys_argv, self._get_config_name(config))
        else:
            output_dir = self.args.output_dir
            if "-" in output_dir:
                unique_name = os.path.basename(output_dir).split("-")[0]

        output_dir = os.path.realpath(output_dir)
        output_dir = shorten_file_name(output_dir, FILE_NAME_MAX_LENGTH)

        return output_dir, unique_name


@dataclass
class CkptConfigHandler(ConfigHandler):
    resume: bool = True

    def _obtain_config(self):
        import torch

        path = self.args.path
        model_dict = torch.load(path, map_location="cpu")
        config = cfg_load_preparation(model_dict["hyper_parameters"]["cfg"])
        return config

    def _edit_config(self, config):
        super()._edit_config(config)

        config["global"]["ckpt_path"] = self.args.path
        if self.is_eval:
            config["metadata"].pop("queue_id", None)
            config["metadata"].pop("queue_name", None)
            config["metadata"].pop("queue_eid", None)

    def _get_config_name(self, config):
        config_name_suffix = super()._get_config_name(config)
        if len(config_name_suffix) > 0:
            return "-".join(
                [config["metadata"].get("unique_name", config["metadata"]["config_name"]), config_name_suffix]
            )
        else:
            return config["metadata"].get("unique_name", config["metadata"]["config_name"])


@dataclass
class FileConfigHandler(ConfigHandler):
    def _get_config_name(self, config):
        config_name = None
        if os.path.realpath(TEMPLATES_DIR) != os.path.commonpath(
            [os.path.realpath(TEMPLATES_DIR), os.path.realpath(self.args.path)]
        ):
            config_name = Path(self.args.path).stem

        config_name_suffix = super()._get_config_name(config)

        if config_name is not None:
            if len(config_name_suffix) > 0:
                config_name = "-".join([config_name, config_name_suffix])
        else:
            config_name = config_name_suffix

        return config_name


@dataclass
class JsonnetConfigHandler(FileConfigHandler):
    def _obtain_config(self):
        # actual config is obtained via unify
        config_path = self.args.path
        return config_path


@dataclass
class JsonConfigHandler(FileConfigHandler):
    def _obtain_config(self):
        with open(self.args.path, mode="r") as f:
            config = json.load(f)
        return config


def get_last_config_index(output_dir="."):
    if not os.path.isfile(os.path.join(output_dir, METADATA_PATH, "config.json")):
        return -1

    run_index = 1
    while True:
        if os.path.exists(os.path.join(output_dir, METADATA_PATH, str(run_index))):
            run_index += 1
        else:
            run_index -= 1
            break
    return run_index


def directory_config_handler(**kwargs):
    args = kwargs["args"]

    if os.path.isdir(args.path):
        ckpt_path = os.path.join(args.path, "checkpoints", "best.ckpt" if kwargs["is_eval"] else "last.ckpt")
        # TODO: P0 - Handle fault tolerant

        base_path = METADATA_PATH
        last_config_index = get_last_config_index(args.path)
        if last_config_index != 0:
            base_path = os.path.join(METADATA_PATH, str(last_config_index))
        metadata_config_path = os.path.join(args.path, base_path, "config.json")

        if os.path.isfile(ckpt_path):
            args.path = ckpt_path
            return CkptConfigHandler(**kwargs)
        else:
            args.path = metadata_config_path
            return JsonConfigHandler(**kwargs, resume=True)
    else:
        raise ValueError(f"{args.path} is not a directory")


def dynamic_checkpoint_selection(name, allow):
    """
    Instead of using the full checkpoint path, use just the experiment name and input the checkpoint name
    """
    # TODO: To be used when the whole config code is replaced with Fire
    if not allow:
        raise ValueError(f"Dynamic choice of checkpoint is not allowed")
    else:
        folder_names = [folder for folder in os.listdir(PARENT_OUTPUT_DIR) if folder.startswith(name)]
        if len(folder_names) == 1:
            folder_name = folder_names[0]
        else:
            raise ValueError(f"Found {len(folder_names)} folders starting with name {name}. Be more specific.")
        if os.path.isdir(os.path.join(PARENT_OUTPUT_DIR, folder_name)):
            file_name = ""
            print("    ".join(os.listdir(os.path.join(PARENT_OUTPUT_DIR, folder_name, "checkpoints"))))
            while len(file_name) == 0:
                file_name = input("Input file name:\n").strip()
            return os.path.join(PARENT_OUTPUT_DIR, folder_name, "checkpoints", file_name)
        else:
            raise ValueError(f"Couldn't find {name} as an output directory")


EXTENSION_TO_CONFIG_HANDLER = {
    None: directory_config_handler,
    ".ckpt": CkptConfigHandler,
    ".jsonnet": JsonnetConfigHandler,
    ".json": JsonConfigHandler,
}


def get_wandb_logger_config(config, return_index=False):
    wandb_logger = None
    wandb_logger_index = -1
    loggers = config["trainer"].get("logger", [])
    if isinstance(loggers, Iterable):
        for i, metrics_logger in enumerate(loggers):
            if "wandb" in metrics_logger["_target_"].lower():
                wandb_logger = metrics_logger
                wandb_logger_index = i
                break
    return (wandb_logger, wandb_logger_index) if return_index else wandb_logger


def replace_value_recursive(obj, old, new):
    if isinstance(obj, str):
        return obj.replace(old, new)

    if isinstance(obj, Mapping):
        for key, value in obj.items():
            obj[key] = replace_value_recursive(value, old, new)
    elif isinstance(obj, Sequence):
        for i, value in enumerate(obj):
            obj[i] = replace_value_recursive(value, old, new)
    return obj


def replace_key_recursive(obj, old, new):
    if isinstance(obj, Mapping):
        for value in obj.values():
            replace_key_recursive(value, old, new)
        if old in obj:
            obj[new] = obj.pop(old)
    elif isinstance(obj, Sequence) and not isinstance(obj, str):
        for i, value in enumerate(obj):
            obj[i] = replace_key_recursive(value, old, new)
    return obj


def unify(config: Union[str, Dict], unify_files: List[str], parsed_jsonnet_overrides: List[Override]):
    def value_to_json_literal(value):
        if isinstance(value, str):
            value = shlex.quote(value)
            if not ('"' in value or "'" in value):
                value = f'"{value}"'
            return value
        elif isinstance(value, bool):
            return str(value).lower()
        else:
            return value

    def get_dict_str(d, key_suffix, hidden=False):
        colon = "::" if hidden else ":"
        content = []
        for key, value in d.items():
            if isinstance(value, dict):
                value_str = get_dict_str(value, key_suffix)
            else:
                value_str = value_to_json_literal(value)
            content.append(f"\"{key}\"{key_suffix if isinstance(value, dict) else ''}{colon} {value_str}")
        return f"{{{', '.join(content)}}}"

    ext_vars = _environment_variables()  # accessible through std.extVar(VAR_NAME)

    initial_config_str = ""
    unify_files = list(unify_files)
    if isinstance(config, str):
        config_path = config
        unify_files.insert(0, config_path)
    elif isinstance(config, dict):
        initial_config_str = json.dumps(config)

    unified_jsonnet_lines = []
    for i, unify_file in enumerate(unify_files):
        unified_jsonnet_lines.append(f'local var_{i} = import "{unify_file}";')

    unify_imports_str = "+".join(f"var_{i}" for i in range(len(unify_files)))

    unify_overrides = []
    for parsed_override in parsed_jsonnet_overrides:
        # TODO: support dot expansion + selection of :, ::, +: or +:: (use regex to match for j[+:]+/) but without lists indexing (because it's not possible)
        # Reasonable: always assume +:
        d = {}
        key = parsed_override.key_or_group
        set_dot(d, key, parsed_override.value(), force=True)
        unify_overrides.append(get_dict_str(d, key_suffix="+", hidden=key in HIDDEN_KEYS))

    unify_overrides_str = "+".join(unify_overrides)

    unified_jsonnet_lines.append(
        "+".join([s for s in [initial_config_str, unify_imports_str, unify_overrides_str] if len(s) > 0])
    )

    with open(os.path.join(TEMPLATES_DIR, "empty.json"), mode="r") as f:
        empty = json.load(f)

    unified_jsonnet_str = "\n".join([s for s in unified_jsonnet_lines if len(s) > 0])
    try:
        unified_config = json.loads(
            evaluate_snippet(os.path.join(".", "unified.jsonnet"), unified_jsonnet_str, ext_vars=ext_vars)
        )
    except Exception as e:
        print(unified_jsonnet_str)
        raise e

    config = unified_config

    for key in HIDDEN_KEYS:
        config.pop(key, None)

    return config


def merge(config: Dict, merge_files, override=True):
    ext_vars = _environment_variables()  # accessible through std.extVar(VAR_NAME)

    for merge_file in merge_files:
        with open(merge_file, mode="r") as f:
            merge_str = f.read()
        merge_dict = json.loads(evaluate_snippet(merge_file, merge_str, ext_vars=ext_vars))
        config = (
            with_fallback(preferred=merge_dict, fallback=config)
            if override
            else with_fallback(preferred=config, fallback=merge_dict)
        )

    return config


MANDATORY_VALUE = "<??>"


def verify_mandatory_fields(config, path=None, mandatory_fields=None, prev_type=None):
    path_prefix = "" if path is None else path + "."
    first_call = False
    if mandatory_fields is None:
        mandatory_fields = []
        first_call = True
    if isinstance(config, Mapping):
        target = config.get("_target_")

        for key, value in config.items():
            should_add_target = target is not None and key != "_target_" and prev_type == "list"
            verify_mandatory_fields(
                value,
                path_prefix + key + (f" (_target_={target})" if should_add_target else ""),
                mandatory_fields,
                "dict",
            )
    elif isinstance(config, Sequence) and not isinstance(config, str):
        for i, value in enumerate(config):
            verify_mandatory_fields(value, path_prefix + str(i), mandatory_fields, "list")
    else:
        if config == MANDATORY_VALUE:
            mandatory_fields.append(path)

    if first_call:
        error_message = f"Please override these {len(mandatory_fields)} mandatory fields:"
        mandatory_fields.sort()
        for mandatory_field in mandatory_fields:
            error_message += f"\n{mandatory_field}"
        if len(mandatory_fields) > 0:
            print(error_message)
            exit()


def cfg_save_preparation(cfg: DictConfig, recursive=False):
    if not recursive:
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg)

    if isinstance(cfg, dict):
        if "_target_" in cfg:
            cfg["__target__"] = cfg["_target_"]
            del cfg["_target_"]

        for v in cfg.values():
            if isinstance(v, dict) or isinstance(v, list):
                cfg_save_preparation(v, recursive=True)
    elif isinstance(cfg, list):
        for v in cfg:
            cfg_save_preparation(v, recursive=True)

    if not recursive:
        return cfg


def cfg_load_preparation(cfg: Dict[str, Any], recursive=False):
    if not recursive:
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg)

    if isinstance(cfg, dict):
        if "__target__" in cfg:
            cfg["_target_"] = cfg["__target__"]
            del cfg["__target__"]

        for v in cfg.values():
            if isinstance(v, dict) or isinstance(v, list):
                cfg_load_preparation(v, recursive=True)
    elif isinstance(cfg, list):
        for v in cfg:
            cfg_load_preparation(v, recursive=True)

    if not recursive:
        return cfg


def safe_open_dict(cfg):
    return open_dict(cfg) if isinstance(cfg, DictConfig) else contextlib.nullcontext()
