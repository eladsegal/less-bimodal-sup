import os
import time
from src.utils.logging import set_logger
import logging

set_logger()
import fire
import subprocess

import wandb

import logging

logger = logging.getLogger(__name__)


class WandbDiff:
    def __init__(self, wandb_id_1, wandb_id_2, entity, project):
        self.wandb_id_1 = wandb_id_1
        self.wandb_id_2 = wandb_id_2
        self.entity = entity
        self.project = project

    def run(self):
        wandb_id_1 = self.wandb_id_1
        wandb_id_2 = self.wandb_id_2

        output_dir_1 = get_output_dir(wandb_id_1)
        self.clone_git_state(wandb_id_1, self.entity, self.project, output_dir_1)

        output_dir_2 = get_output_dir(wandb_id_2)
        self.clone_git_state(wandb_id_2, self.entity, self.project, output_dir_2)

        output_dir = get_output_dir(f"{wandb_id_1}_{wandb_id_2}")
        self.create_diff_view(output_dir_1, output_dir_2, output_dir)

    def create_diff_view(self, output_dir_1, output_dir_2, output_dir):
        logger.info(f"Creating a diff view in {output_dir}")
        for dir_ in [output_dir_1, output_dir_2]:
            execute(f"cp -a {os.path.join(dir_, '.')} {output_dir}", retries=10)
            execute(f"rm -rf {os.path.join(output_dir, '.git')}", retries=10)
        execute(f"cp -r {os.path.join(output_dir_1, '.git')} {output_dir}", retries=10)


def execute(command, cwd=None, retries=0):
    logger.info(f"$ {command}")
    while True:
        process = subprocess.run(command.split(), cwd=cwd, capture_output=True)
        p_out = process.stdout.strip().decode("utf-8")
        p_err = process.stderr.strip().decode("utf-8")
        if process.returncode != 0:
            if retries == 0:
                raise Exception(p_err)
            else:
                retries -= 1
                time.sleep(1)
        else:
            return p_out


def clone_git_state(wandb_id, entity, project, output_dir):
    logger.info(f"Cloning git state for {wandb_id} in {output_dir}")
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    import wandb

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{wandb_id}")

    commit = run.commit
    git_url = execute("git config --get remote.origin.url")
    execute(f"git clone {git_url} {output_dir}")

    original_cwd = os.getcwd()
    os.chdir(output_dir)

    execute(f"git reset --hard {commit}", retries=10)

    run.file("metadata/git_diff.patch").download()
    execute("git apply metadata/git_diff.patch")
    run.file("metadata/untracked_files.zip").download()
    execute("unzip metadata/untracked_files.zip")

    execute(f"rm -rf metadata", retries=10)

    execute(f"git add -A")
    execute(f'git commit -m "{wandb_id}"')

    os.chdir(original_cwd)


def get_output_dir(wandb_id=""):
    return os.path.join("..", "experiments", wandb_id)


def diff(wandb_id_1, wandb_id_2, entity, project):
    WandbDiff(wandb_id_1, wandb_id_2, entity, project).run()


def get_code_version(wandb_id, entity, project):
    api = wandb.Api()
    os.makedirs(get_output_dir(), exist_ok=True)

    output_dir_1 = get_output_dir(wandb_id)
    clone_git_state(wandb_id, entity, project, output_dir_1)


if __name__ == "__main__":
    fire.Fire(
        {
            "diff": diff,
            "code": get_code_version,
        }
    )
