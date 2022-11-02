import socket
import os

from pytorch_lightning.utilities import rank_zero_only

from scripts.utils import is_on_slurm

from src.utils.config import safe_open_dict


@rank_zero_only
def execution_info_config_modifications(cfg):
    with safe_open_dict(cfg):
        if is_on_slurm() and os.environ.get("SLURM_JOB_NAME") != "cpu_copy_files.sh":
            cfg["metadata"]["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
        else:
            cfg["metadata"]["pid"] = os.getpid()
        cfg["metadata"]["hostname"] = socket.gethostname()
