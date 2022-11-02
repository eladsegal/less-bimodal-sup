from typing import Dict, Any
import argparse
import os
import sys
import shutil

sys.path.insert(0, os.getcwd())
import time
import runpy
import importlib
import subprocess
import json
import random
import string
import copy
import tempfile
import subprocess

from omegaconf import OmegaConf
from hydra._internal.config_loader_impl import ConfigLoaderImpl

from scripts.utils import (
    has_slurm,
    get_sys_argv,
    prep_command,
    get_parsed_overrides,
    handle_args_to_delete_or_null,
)


def main(command_dict: Dict[str, Any], dummy=False, no_slurm=False, slurm_config_path=None):
    command: str = command_dict["command"]

    sys_argv = get_sys_argv(command)
    program = sys_argv[0]
    # Find the module name
    if not program.endswith(".py"):
        program = program.replace(".", "/") + ".py"
    module_name = program.replace("/", ".")[: -len(".py")]
    if os.path.isdir(os.path.dirname(program)):
        sys.path.insert(0, os.path.dirname(program))

    if has_slurm() and not no_slurm:
        output_dir = command_dict["output_dir"]

        slurm_config = {
            "job-name": os.path.basename(os.path.abspath(output_dir)),
            "output": os.path.join(output_dir, "out.txt"),
            "error": os.path.join(output_dir, "err.txt"),
            "partition": "killable",
            # "time": 10,
            "nodes": 1,  # If we don't use PyTorch Lightning's SlurmEnvironment ("ntasks-per-node"!="gpus-per-node"), then we need to define MASTER_ADDR and MASTER_PORT for multi-node training
            "mem": 50000,
            "cpus-per-task": 6,
            "gpus-per-node": 0,
            "signal": "USR1@90",
            "open-mode": "append",
            # "exclude": "rack-bgw-dgx1,rack-gww-dgx1",
            "constraint": '"geforce_rtx_3090|a5000|a6000"',  # |tesla_v100|a100_sxm_80gb|quadro_rtx_8000
            # "nodelist": "rack-bgw-dgx1",
        }
        if slurm_config_path is not None:
            with open(slurm_config_path) as f:
                slurm_config.update(json.load(f))

        # Merge slurm_config with slurm command line args
        # Replacing "-"" with "_" because hydra doesn't support args with "-"
        slurm_config = {k.replace("-", "_"): v for k, v in slurm_config.items()}
        slurm_config.update({k.replace("-", "_"): v for k, v in command_dict["slurm_config"].items()})
        slurm_config = OmegaConf.create(slurm_config)
        parsed_overrides = get_parsed_overrides(
            [
                arg[len("slurm.") : arg.index("=")].replace("-", "_") + arg[arg.index("=") :]
                for arg in sys_argv
                if arg.startswith("slurm.") and "=" in arg
            ]
        )
        ConfigLoaderImpl._apply_overrides_to_config(parsed_overrides, slurm_config)
        slurm_config = OmegaConf.to_container(slurm_config)
        slurm_config = {k.replace("_", "-"): v for k, v in slurm_config.items()}
        slurm_config_copy = copy.deepcopy(slurm_config)
        handle_args_to_delete_or_null(slurm_config)

        if "ntasks-per-node" not in slurm_config_copy:
            # Can be 1 after PyTorchLightning/pytorch-lightning/pull/11406 is merged,
            # which will make use of LightningEnvironment instead of SlurmEnvironment
            slurm_config["ntasks-per-node"] = max(1, slurm_config.get("gpus-per-node", 0))

        # sbatch_file = f"/tmp/temp_{''.join(random.SystemRandom().choice(string.ascii_lowercase + string.digits) for _ in range(5))}.sbatch"
        os.makedirs("../slurm", exist_ok=True)
        output_folder = os.path.basename(output_dir)
        sbatch_file = os.path.join("../slurm", output_folder[: output_folder.index("-")] + ".sbatch")
        contents = []
        contents.append("#!/bin/sh\n")
        for key, value in slurm_config.items():
            contents.append(f"#SBATCH --{key}{f'={value}' if value != '' else ''}\n")

        contents.append("set -e\n")  # terminate if any of the lines fail

        # TODO: find a better way to add this so execute.py will be as general as possible.
        if slurm_config["ntasks-per-node"] > 1:
            if program in ["src/train.py", "src/validate.py", "src/test.py"]:
                contents.append(
                    f"srun --gpus-per-node=0 --ntasks-per-node=1 --ntasks=1 --nodes=1 {command} global.only_preprocess_data=true trainer.gpus=0\n"
                )

        contents.append(f"srun {command}\n")

        with open(sbatch_file, mode="w", encoding="utf=8") as f:
            for content in contents:
                f.write(content)

        print()
        with open(sbatch_file, mode="r", encoding="utf=8") as f:
            print(f.read())
        print(sbatch_file)

        # subprocess.run(f"sbatch --test-only {sbatch_file}".split(), check=True)
        if not dummy:
            process = subprocess.run(f"sbatch {sbatch_file}", shell=True, check=True, capture_output=True)
            p_out = process.stdout.strip().decode("utf-8")

            os.makedirs(output_dir, exist_ok=True)
            metadata_path = os.path.join(output_dir, "metadata")
            os.makedirs(metadata_path, exist_ok=True)
            with open(os.path.join(metadata_path, "slurm_config.json"), mode="w") as f:
                json.dump(slurm_config, f, indent=4)
    else:
        os.environ.pop("SLURM_NTASKS", None)  # makes sure PyTorch Lightning doesn't use Slurm
        sys.argv = sys_argv

        print()
        print(" ".join((["python"] if command.startswith("python") else []) + sys.argv))
        if not dummy:
            module_spec = importlib.util.find_spec(module_name)
            module_found = module_spec is not None
            if module_found:
                runpy.run_module(module_name, run_name="__main__")
            else:
                subprocess.run(sys_argv, check=True)


def argparse_common(parser):
    parser.add_argument("--dummy", action="store_true", default=False)
    parser.add_argument("--no_slurm", action="store_true", default=False)
    parser.add_argument("--slurm_config", type=str, help="path to a .json file with slurm config")
    parser.add_argument("--now", action="store_true", default=False)


def argparse_commands_script():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("command_path", type=str)
    parser.add_argument("id", type=str)
    argparse_common(parser)
    args, unknown = parser.parse_known_args()

    command_module = args.command_path.replace("/", ".")[: -len(".py")]
    command_dict = importlib.import_module(command_module).get_command(args.id, unknown, args)

    return args, command_dict


def argparse_inline():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    argparse_common(parser)
    args, unknown = parser.parse_known_args()

    slurm_config = {}
    command_dict = prep_command([], slurm_config, unknown, args)

    return args, command_dict


if __name__ == "__main__":
    # os.environ["PL_FAULT_TOLERANT_TRAINING"] = os.environ.get("PL_FAULT_TOLERANT_TRAINING", "1")
    assert len(sys.argv) > 1

    if sys.argv[1].endswith(".py"):
        args, command_dict = argparse_commands_script()
    else:
        args, command_dict = argparse_inline()

    main(command_dict, dummy=args.dummy, no_slurm=args.no_slurm, slurm_config_path=args.slurm_config)
