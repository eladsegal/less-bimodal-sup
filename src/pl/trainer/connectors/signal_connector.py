import logging
import os
import sys
from signal import Signals
import subprocess
from types import FrameType
import subprocess

import torch

from pytorch_lightning.trainer.connectors.signal_connector import SignalConnector

from utils.general import resolve_relative_paths

log = logging.getLogger(__name__)


class MySignalConnector(SignalConnector):
    def slurm_sigusr1_handler_fn(self, signum: Signals, frame: FrameType) -> None:
        self.handle_preemption(explicit_requeue=True, message="handling SIGUSR1")

    def sigterm_handler_fn(self, signum: Signals, frame: FrameType) -> None:
        self.handle_preemption(explicit_requeue=False, message="handling sigterm")

    def handle_preemption(self, *, explicit_requeue, message):
        if os.path.isfile("preempted"):
            return
        log.info(message)

        # save logger to make sure we get all the metrics
        for logger in self.trainer.loggers:
            logger.finalize("finished")

        use_barrier = torch.distributed.is_available() and torch.distributed.is_initialized()

        if self.trainer.is_global_zero:
            with open("preempted", mode="w") as f:
                f.write(os.getcwd())

            if explicit_requeue:
                # find job id
                job_id = os.environ["SLURM_JOB_ID"]
                slurm_config_path = os.path.realpath(os.path.join(os.getcwd(), "metadata", "slurm_config.json"))
                last_ckpt_path = os.path.realpath(os.path.join(os.getcwd(), "checkpoints", "last.ckpt"))
                if os.path.isfile(last_ckpt_path):
                    LIGHTNING_REQUEUE = True
                    if LIGHTNING_REQUEUE:
                        cmd = ["scontrol", "requeue", job_id]

                        # requeue job
                        log.info(f"requeing job {job_id}...")
                        try:
                            process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                        except FileNotFoundError:
                            # This can occur if a subprocess call to `scontrol` is run outside a shell context
                            # Re-attempt call (now with shell context). If any error is raised, propagate to user.
                            # When running a shell command, it should be passed as a single string.
                            joint_cmd = [str(x) for x in cmd]
                            process = subprocess.run(
                                " ".join(joint_cmd), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
                            )
                        p_out = process.stdout.strip().decode("utf-8")
                        log.info(p_out)
                        if process.returncode == 0:
                            log.info(f"requeued exp {job_id}")
                        else:
                            log.warning("requeue failed...")
                    else:
                        cmd = f"python scripts/execute.py python src/train.py {last_ckpt_path} --slurm_config {slurm_config_path}"

                        log.info(f"requeing job...")
                        prev_working_directory = os.getcwd()
                        os.chdir(resolve_relative_paths("."))
                        process = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                        p_out = process.stdout.strip().decode("utf-8")
                        log.info(p_out)
                        if process.returncode == 0:
                            log.info(f"requeued exp {job_id}")
                        else:
                            log.info("requeue failed...")
                        os.chdir(prev_working_directory)
                else:
                    log.info("No last.ckpt found. No point in requeueing without increasing the allotted time.")

            self.trainer.upload_files(["out.log"])

            # close experiment to avoid issues
            if self.trainer.logger:
                self.trainer.logger.finalize("finished")

            if use_barrier:
                torch.distributed.barrier()

        if use_barrier and not self.trainer.is_global_zero:
            torch.distributed.barrier()

        sys.exit()
