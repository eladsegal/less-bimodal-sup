import os
import GPUtil
import shutil
import subprocess
import socket

import json
from omegaconf import DictConfig, open_dict
import torch
from filelock import FileLock

from utils.general import resolve_relative_paths, pid_exists


def take_free_gpus(num_gpus, ignore_used=False):
    taken_file_path = _get_taken_file_path()
    os.makedirs(os.path.dirname(taken_file_path), exist_ok=True)

    """if num_gpus > 0 and "SLURM_PROCID" in os.environ and len(os.environ.get("CUDA_VISIBLE_DEVICES", "")) == 0:
        raise Exception("GPU was not assigned on SLURM")"""
    if "LOCAL_RANK" in os.environ or "SLURM_PROCID" in os.environ:
        # if in DDP we have "LOCAL_RANK"/"SLURM_PROCID" then the right CUDA_VISIBLE_DEVICES was already set
        # cpu_copy_files.sh is a disguise to vscode.sh
        if os.environ.get("SLURM_JOB_NAME") != "cpu_copy_files.sh":
            return

    CUDA_VISIBLE_DEVICES_STR = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    if len(CUDA_VISIBLE_DEVICES_STR.strip()) == 0:
        CUDA_VISIBLE_DEVICES_STR = "0,1,2,3,4,5,6,7"
    CUDA_VISIBLE_DEVICES = list(map(int, CUDA_VISIBLE_DEVICES_STR.split(",")))

    if os.environ.get("CUDA_DEVICES_CHOSEN", "0") == "1":
        # only update taken gpus
        with FileLock(f"{taken_file_path}.lock") as lock:
            taken_gpus, taken_gpus_dict = _get_taken_gpus()
            taken_gpus_dict.update(
                {str(os.getpid()): list(set(taken_gpus_dict.get(str(os.getpid()), []) + CUDA_VISIBLE_DEVICES))}
            )
            with open(taken_file_path, mode="w") as f:
                json.dump(taken_gpus_dict, f)
        return

    with FileLock(f"{taken_file_path}.lock") as lock:
        taken_gpus, taken_gpus_dict = _get_taken_gpus()

        all_gpus = GPUtil.getGPUs()
        available_gpus = [gpu for gpu in all_gpus if gpu.id in CUDA_VISIBLE_DEVICES and gpu.id not in taken_gpus]
        free_gpus_ids = [
            gpu.id
            for gpu, availability in zip(
                available_gpus, GPUtil.getAvailability(available_gpus, maxLoad=0.2, maxMemory=0.01)
            )
            if bool(availability) or ignore_used
        ]
        selected_gpus = free_gpus_ids[:num_gpus]

        if len(all_gpus) > 0 and len(free_gpus_ids) < num_gpus:
            raise GPUException(
                f"Requested {num_gpus} GPUs but only {len(free_gpus_ids)} are available: {free_gpus_ids}"
            )

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))

        taken_gpus_dict.update(
            {str(os.getpid()): list(set(taken_gpus_dict.get(str(os.getpid()), []) + selected_gpus))}
        )

        with open(taken_file_path, mode="w") as f:
            json.dump(taken_gpus_dict, f)


def _get_taken_file_path():
    taken_file_path = resolve_relative_paths(os.path.join("../taken_gpus", f"{socket.gethostname()}.json"))
    return taken_file_path


def _get_taken_gpus():
    taken_file_path = _get_taken_file_path()
    taken_gpus = set()
    if os.path.isfile(taken_file_path):
        with open(taken_file_path, mode="r") as f:
            taken_gpus_dict = json.load(f)
        pids_to_delete = set()
        for pid_str in taken_gpus_dict.keys():
            pid = int(pid_str)
            if pid_exists(pid):
                taken_gpus.update(taken_gpus_dict[pid_str])
            else:
                pids_to_delete.add(pid_str)
        for pid_to_delete in pids_to_delete:
            del taken_gpus_dict[pid_to_delete]
    else:
        taken_gpus_dict = {}
    return taken_gpus, taken_gpus_dict


def remove_gpus_for_pid(gpus, pid):
    taken_file_path = _get_taken_file_path()
    with FileLock(f"{taken_file_path}.lock") as lock:
        taken_gpus, taken_gpus_dict = _get_taken_gpus()
        if str(pid) in taken_gpus_dict:
            taken_gpus_for_pid = taken_gpus_dict[str(pid)]
            for gpu in gpus:
                if gpu in taken_gpus_for_pid:
                    taken_gpus_for_pid.remove(gpu)
            with open(taken_file_path, mode="w") as f:
                json.dump(taken_gpus_dict, f)


def get_device_memory_info(device):
    queries = ["used", "total"]
    gpu_memory_queries = {}
    for query in queries:
        nvidia_smi_path = shutil.which("nvidia-smi")
        if nvidia_smi_path is None:
            raise FileNotFoundError("nvidia-smi: command not found")
        result = subprocess.run(
            [nvidia_smi_path, f"--query-gpu=memory.{query}", "--format=csv,nounits,noheader"],
            encoding="utf-8",
            capture_output=True,
            check=True,
        )
        # Convert lines into a dictionary
        gpu_memory_queries[query] = [float(x) for x in result.stdout.strip().split(os.linesep)]

    return {query: gpu_memory_queries[query][device] for query in queries}


def config_modifications_for_gpu(cfg: DictConfig):
    if "gpus" not in cfg.trainer:
        with open_dict(cfg):
            cfg.trainer.gpus = 0
    if cfg.trainer.gpus != 0:
        take_free_gpus(cfg.trainer.gpus, ignore_used=cfg["global"].get("ignore_used_gpus", False))
        if not torch.cuda.is_available():
            with open_dict(cfg):
                cfg.trainer.gpus = 0
                cfg.datamodule.pin_memory = False

    if cfg.trainer.get("precision") == 16 and cfg.trainer.gpus == 0:
        # Without this precision will be converted by PyTorch Lightning to bf16 automatically which is extremely slow on CPU
        with open_dict(cfg):
            del cfg.trainer["precision"]


class GPUException(Exception):
    pass
