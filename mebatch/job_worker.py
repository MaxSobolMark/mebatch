"""
A job worker is a process that runs on every machine that will execute jobs.
The orchestrator will send jobs to the job worker, and the job worker will execute them in parallel.

The orchestrator and the job worker communicate through the file system.
Jobs are assigned at the file {mebatch_dir}/job_pools/active_pools/{id}/new_jobs.txt.

"""
from typing import List
import click
import time
import subprocess
from mebatch.job_pool import MonitoredProcessPoolExecutor, GracefulKiller
from mebatch.slack import send_slack_message
import tensorflow as tf  # For GCS support.
from filelock import FileLock  # To lock the job queue file.
from gcs_mutex_lock import gcs_lock


class GCSFileLock:
    def __init__(self, file_path: str):
        self.file_path = file_path
        assert file_path.startswith("gs://"), "The file path must start with gs://."

    def __enter__(self):
        acquired = gcs_lock.wait_for_lock_expo(self.file_path)
        assert acquired, f"Could not acquire lock for {self.file_path}."

    def __exit__(self, exc_type, exc_val, exc_tb):
        gcs_lock.unlock(self.file_path)


def run_one_job(
    job_name: str,
    command: str,
    send_slack_messages: bool,
):
    process = subprocess.Popen(
        command,
        shell=True,
    )
    error_code = process.wait()
    # Send a slack message alerting that the job finished.
    if send_slack_messages:
        if error_code == 0:
            send_slack_message(
                f"🟢 Job {job_name} finished successfully.",
            )
        else:
            send_slack_message(
                f"🔴 Job {job_name} finished with error code {error_code}.",
            )


def run_new_jobs(
    executor: MonitoredProcessPoolExecutor,
    new_jobs_file_path: str,
    new_jobs_file_lock: FileLock,
    send_slack_messages: bool = True,
    is_tpu: bool = False,
    tpu_availabilities: List[bool] = None,
):
    with new_jobs_file_lock:
        with open(new_jobs_file_path, "r") as f:
            new_jobs = f.read().splitlines()
        if not new_jobs:
            return
        if not is_tpu:
            for job in new_jobs:
                job_name, command = job.split("\t")
                executor.submit(
                    run_one_job,
                    job_name,
                    command,
                    send_slack_messages,
                )
            # Clear the new jobs file.
            with tf.io.gfile.GFile(new_jobs_file_path, "w") as f:
                f.write("")
        else:
            num_jobs_submitted = 0
            for job in new_jobs:
                job_name, command, num_tpus = job.split("\t")
                num_tpus = int(num_tpus)
                assert num_tpus in [1, 2, 4], "Invalid number of TPUs."
                if num_tpus == 1:
                    free_tpu = tpu_availabilities.index(True)
                    if free_tpu == -1:
                        break
                    tpu_availabilities[free_tpu] = False
                    callback = lambda future: tpu_availabilities[free_tpu] = True
                    executor.submit(
                        run_one_job,
                        job_name,
                        f"TPU{free_tpu} {command}",
                        send_slack_messages,
                    ).add_done_callback(callback)
                    num_jobs_submitted += 1
                elif num_tpus == 2:
                    # either 01 or 23
                    free_tpu = -1
                    for i in range(0, 4, 2):
                        if tpu_availabilities[i] and tpu_availabilities[i + 1]:
                            free_tpu = i
                            break
                    if free_tpu == -1:
                        break
                    tpu_availabilities[free_tpu] = False
                    tpu_availabilities[free_tpu + 1] = False
                    callback = lambda future: (
                        tpu_availabilities[free_tpu] = True,
                        tpu_availabilities[free_tpu + 1] = True,
                    )
                    executor.submit(
                        run_one_job,
                        job_name,
                        f"TPU{free_tpu}{free_tpu + 1} {command}",
                        send_slack_messages,
                    ).add_done_callback(callback)
                    num_jobs_submitted += 1
                elif num_tpus == 4:
                    if not all(tpu_availabilities):
                        break
                    for i in range(4):
                        tpu_availabilities[i] = False
                    callback = lambda future: (
                        tpu_availabilities[0] = True,
                        tpu_availabilities[1] = True,
                        tpu_availabilities[2] = True,
                        tpu_availabilities[3] = True,
                    )
                    executor.submit(
                        run_one_job,
                        job_name,
                        command,
                        send_slack_messages,
                    ).add_done_callback(callback)
                    num_jobs_submitted += 1
            # Update the new jobs file with the remaining jobs.
            with tf.io.gfile.GFile(new_jobs_file_path, "w") as f:
                f.write("\n".join(new_jobs[num_jobs_submitted:]))
                
                        


@click.command()
@click.argument("mebatch_dir", type=str)
@click.argument("id", type=str)
@click.argument("max_num_parallel_jobs", type=int)
@click.option(
    "--send-slack-messages/--no-send-slack-messages",
    default=True,
    help="Whether to send slack messages.",
)
@click.option(
    "--is-tpu/--no-is-tpu",
    default=False,
    help="Whether the job worker is running on a TPU.",
)
def job_worker(
    mebatch_dir: str,
    id: str,
    max_num_parallel_jobs: int,
    send_slack_messages: bool = True,
    is_tpu: bool = False,
):
    assert max_num_parallel_jobs > 0, "max_num_parallel_jobs must be positive."
    new_jobs_file_path = f"{mebatch_dir}/job_pools/active_pools/{id}/new_jobs.txt"
    new_jobs_file_lock_path = (
        f"{mebatch_dir}/job_pools/active_pools/{id}/new_jobs.txt.lock"
    )
    if not tf.io.gfile.exists(new_jobs_file_path):
        with tf.io.gfile.GFile(new_jobs_file_path, "w") as f:
            f.write("")

    new_jobs_file_lock = (
        FileLock(new_jobs_file_lock_path, timeout=5)
        if "gs://" not in new_jobs_file_lock_path
        else GCSFileLock(new_jobs_file_lock_path)
    )

    executor = MonitoredProcessPoolExecutor(
        max_workers=max_num_parallel_jobs,
    )
    killer = GracefulKiller()

    if is_tpu:
        # Assume 4 chips available.
        tpu_availabilities = [True] * 4
    else:
        tpu_availabilities = None

    while True:
        # Run the jobs in new_jobs.txt.
        run_new_jobs(
            executor=executor,
            new_jobs_file_path=new_jobs_file_path,
            new_jobs_file_lock=new_jobs_file_lock,
            pipe_stdout=pipe_stdout,
            pipe_stderr=pipe_stderr,
            send_slack_messages=send_slack_messages,
            is_tpu=is_tpu,
            tpu_availabilities=tpu_availabilities,
        )
        # Check if the job worker should exit.
        if killer.kill_now:
            break
        # Sleep for a while before checking for new jobs.
        time.sleep(5)


if __name__ == "__main__":
    job_worker()
