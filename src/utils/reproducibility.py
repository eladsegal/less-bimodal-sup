import json
import os
import sys
import time
import subprocess
import socket
import shutil
import glob
import zipfile
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd

from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.utilities import rank_zero_only

from src.utils.config import METADATA_PATH, get_last_config_index
from utils.general import resolve_relative_paths

import logging

logger = logging.getLogger(__name__)

IMPORTANT_FILE_PATHS = ["metadata/slurm_config.json"]


def save_git_info(output_dir, repo_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    if repo_dir is None:
        repo_dir = os.getcwd()

    # Get to the top level git dir
    process = subprocess.run("git rev-parse --show-toplevel".split(), cwd=repo_dir, capture_output=True)
    p_out = process.stdout.strip().decode("utf-8")
    p_err = process.stderr.strip().decode("utf-8")
    if process.returncode != 0:
        raise Exception(p_err)
    repo_dir = p_out

    git_instructions_file_path = os.path.join(output_dir, "git_instructions.txt")
    git_diff_file_path = os.path.join(output_dir, "git_diff.patch")
    git_untracked_file_path = os.path.join(output_dir, "untracked_files.zip")

    process = subprocess.run("git rev-parse --verify HEAD".split(), cwd=repo_dir, capture_output=True)
    p_out = process.stdout.strip().decode("utf-8")
    p_err = process.stderr.strip().decode("utf-8")
    if process.returncode != 0:
        raise Exception(p_err)
    commit_hash = p_out

    process = subprocess.run(
        "git ls-files --other --full-name --exclude-standard".split(),
        cwd=repo_dir,
        capture_output=True,
    )
    p_out = process.stdout.strip().decode("utf-8")
    p_err = process.stderr.strip().decode("utf-8")
    if process.returncode != 0:
        raise Exception(p_err)
    with zipfile.ZipFile(git_untracked_file_path, "w", zipfile.ZIP_DEFLATED) as f:
        for file_path in [x.strip() for x in p_out.split("\n") if len(x.strip()) > 0]:
            actual_file_path = os.path.join(repo_dir, file_path)
            if os.path.getsize(actual_file_path) > 1024 ** 2:
                logger.info(f"Git saving: Untracked file {file_path} is over 1MB, skipping")
            else:
                f.write(actual_file_path, file_path)

    with open(git_instructions_file_path, mode="w") as f:
        f.writelines(
            [
                "To restore the code, use the following commands in the repository (-b <new_branch_name> is optional):\n",
                f"git checkout -b <new_branch_name> {commit_hash}\n",
                f"git apply {git_diff_file_path}\n",
                f"unzip {git_untracked_file_path}",
            ]
        )
    with open(git_diff_file_path, mode="w") as f:
        subprocess.run("git diff HEAD --binary".split(), cwd=repo_dir, stdout=f, check=True)

    IMPORTANT_FILE_PATHS.extend([git_instructions_file_path, git_diff_file_path, git_untracked_file_path])


@rank_zero_only
def save_files(cfg: DictConfig):
    base_path = METADATA_PATH
    last_config_index = get_last_config_index()
    last_config_index += 1

    if last_config_index != 0:
        base_path = os.path.join(base_path, str(last_config_index))
    os.makedirs(base_path, exist_ok=True)

    # Save config as json
    json_config_path = os.path.join(base_path, "config.json")
    with open(json_config_path, mode="w") as f:
        json.dump(OmegaConf.to_container(cfg), f, indent=4)
    IMPORTANT_FILE_PATHS.append(json_config_path)

    # Save git info
    save_git_info(base_path, repo_dir=get_original_cwd())

    # Save pip freeze and export conda env
    pip_freeze_path = os.path.join(base_path, "pip_freeze.txt")
    with open(pip_freeze_path, mode="w") as f:
        subprocess.run("pip freeze".split(), stdout=f, check=True)
    IMPORTANT_FILE_PATHS.append(pip_freeze_path)
    try:
        conda_env_path = os.path.join(base_path, "conda_env.yaml")
        with open(conda_env_path, mode="w") as f:
            subprocess.run("conda env export".split(), stdout=f, check=True)
        IMPORTANT_FILE_PATHS.append(conda_env_path)
    except Exception as e:
        pass

    # Save the hostname
    hostname_file_path = os.path.join(base_path, socket.gethostname())
    open(hostname_file_path, mode="w").close()
    IMPORTANT_FILE_PATHS.append(hostname_file_path)
    logger.info(f"Running on {socket.gethostname()}")

    # Save the command into command.txt
    command = cfg["metadata"].get("command", f"python {' '.join(sys.argv)}")
    command_file_path = "command.txt" if last_config_index == 0 else f"command_{last_config_index}.txt"
    with open(command_file_path, mode="w") as f:
        f.write(command)
    IMPORTANT_FILE_PATHS.append(command_file_path)

    # Log the command to the log of all exectued commands
    with open(os.path.join(get_original_cwd(), "scripts", "commands", "commands.txt"), mode="a") as f:
        f.write(f"{time.strftime('%d/%m/%Y %H:%M:%S')} - {socket.gethostname()} - {command}\r\n")


def reproducibility_init(cfg: DictConfig):
    if os.getcwd() != get_original_cwd():
        save_files(cfg)

    # Seed everything
    seed_everything(cfg["global"]["seed"], workers=True)
