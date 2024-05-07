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
import signal  # to handle slurm terminations, e.g. preemptions.
import concurrent.futures  # For the process pool.
from mebatch.slack import send_slack_message
import tensorflow as tf  # For GCS support.
from filelock import FileLock  # To lock the job queue file.
from .GCS_file_lock import GCSFileLock

TPU_ENVIRONMENT_VARIABLES = {
    "TPU0": "TPU_VISIBLE_DEVICES=0 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8476 TPU_MESH_CONTROLLER_PORT=8476",
    "TPU1": "TPU_VISIBLE_DEVICES=1 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8477 TPU_MESH_CONTROLLER_PORT=8477",
    "TPU2": "TPU_VISIBLE_DEVICES=2 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8478 TPU_MESH_CONTROLLER_PORT=8478",
    "TPU3": "TPU_VISIBLE_DEVICES=3 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8479 TPU_MESH_CONTROLLER_PORT=8479",
    "TPU01": "TPU_VISIBLE_DEVICES=0,1 TPU_CHIPS_PER_HOST_BOUNDS=1,2,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8476 TPU_MESH_CONTROLLER_PORT=8476",
    "TPU23": "TPU_VISIBLE_DEVICES=2,3 TPU_CHIPS_PER_HOST_BOUNDS=1,2,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8478 TPU_MESH_CONTROLLER_PORT=8478",
}


class GracefulKiller:
    """This class is used to handle signals from the OS, e.g. SIGINT, SIGTERM"""

    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True


class MonitoredProcessPoolExecutor(concurrent.futures.ProcessPoolExecutor):
    """A process pool executor that keeps track of how many workers are running."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._running_workers = 0
        self._futures = []

    def submit(self, *args, **kwargs):
        future = super().submit(*args, **kwargs)
        self._running_workers += 1
        future.add_done_callback(self._worker_is_done)
        self._futures.append(future)
        return future

    def _worker_is_done(self, future):
        self._running_workers -= 1
        self._futures.remove(future)

    def get_pool_usage(self):
        return self._running_workers


def run_one_job(
    job_name: str,
    command: str,
    send_slack_messages: bool,
    mebatch_dir: str,
    worker_id: str,
    pipe_stdout: bool = True,
    pipe_stderr: bool = True,
):
    print(f"Running job {job_name}.")

    if pipe_stdout:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        stdout_file_path = tf.io.gfile.join(
            mebatch_dir, "logs", f"{job_name}_{timestamp}.stdout"
        )
        stdout_file = tf.io.gfile.GFile(stdout_file_path, "w")
        stdout_file.write(f"Command: {command}\n\n")
    else:
        stdout_file = None
    if pipe_stderr:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        stderr_file_path = tf.io.gfile.join(
            mebatch_dir, "logs", f"{job_name}_{timestamp}.stderr"
        )
        stderr_file = tf.io.gfile.GFile(stderr_file_path, "w")
        stderr_file.write(f"Command: {command}\n\n")
    else:
        stderr_file = None

    if send_slack_messages:
        response = send_slack_message(f"🏁 Job {job_name} started on {worker_id}.")
        if response:
            slack_message_ts = response["ts"]
        else:
            slack_message_ts = None

    process = subprocess.Popen(
        command,
        shell=True,
        stdout=stdout_file,
        stderr=stderr_file,
    )
    error_code = process.wait()
    if pipe_stdout:
        stdout_file.close()
    if pipe_stderr:
        stderr_file.close()
    # Send a slack message alerting that the job finished.
    if send_slack_messages:
        if error_code == 0:
            send_slack_message(
                f"🟢 Job {job_name} finished successfully on {worker_id}.",
                thread_ts=slack_message_ts,
            )
        else:
            send_slack_message(
                f"🔴 Job {job_name} finished with error code {error_code} on {worker_id}.",
                thread_ts=slack_message_ts,
            )


def run_new_jobs(
    executor: MonitoredProcessPoolExecutor,
    new_jobs_file_path: str,
    new_jobs_file_lock: FileLock,
    mebatch_dir: str,
    worker_id: str,
    send_slack_messages: bool = True,
    is_tpu: bool = False,
    tpu_availabilities: List[bool] = None,
):
    with new_jobs_file_lock:
        with tf.io.gfile.GFile(new_jobs_file_path, "r") as f:
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
                    mebatch_dir,
                    worker_id,
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

                    def callback(future):
                        if future.exception():
                            print(f"Error in job {job_name}: {future.exception()}")
                        nonlocal tpu_availabilities
                        tpu_availabilities[free_tpu] = True

                    job = executor.submit(
                        run_one_job,
                        job_name,
                        f"{TPU_ENVIRONMENT_VARIABLES['TPU' + str(free_tpu)]} {command}",
                        send_slack_messages,
                        mebatch_dir,
                        worker_id,
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

                    def callback(future):
                        if future.exception():
                            print(f"Error in job {job_name}: {future.exception()}")
                        nonlocal tpu_availabilities
                        tpu_availabilities[free_tpu] = True
                        tpu_availabilities[free_tpu + 1] = True

                    executor.submit(
                        run_one_job,
                        job_name,
                        # f"TPU{free_tpu}{free_tpu + 1} {command}",
                        f"{TPU_ENVIRONMENT_VARIABLES['TPU' + str(free_tpu)+str(free_tpu+1)]} {command}",
                        send_slack_messages,
                        mebatch_dir,
                        worker_id,
                    ).add_done_callback(callback)
                    num_jobs_submitted += 1
                elif num_tpus == 4:
                    if not all(tpu_availabilities):
                        break
                    for i in range(4):
                        tpu_availabilities[i] = False

                    def callback(future):
                        if future.exception():
                            print(f"Error in job {job_name}: {future.exception()}")
                        nonlocal tpu_availabilities
                        tpu_availabilities[0] = True
                        tpu_availabilities[1] = True
                        tpu_availabilities[2] = True
                        tpu_availabilities[3] = True

                    executor.submit(
                        run_one_job,
                        job_name,
                        command,
                        send_slack_messages,
                        mebatch_dir,
                        worker_id,
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
    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")
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
            worker_id=id,
            send_slack_messages=send_slack_messages,
            is_tpu=is_tpu,
            tpu_availabilities=tpu_availabilities,
        )
        # Check if the job worker should exit.
        if killer.kill_now:
            break
        # Sleep for a while before checking for new jobs.
        time.sleep(5)
        print(tpu_availabilities)
    if killer.kill_now:
        print("Job pool killed.")
        if send_slack_messages:
            send_slack_message(
                f"Job pool {id} was killed ☠️.",
            )
    else:
        print("Job pool finished.")
        if send_slack_messages:
            send_slack_message(
                f"Job pool {id} finished 📈.",
            )


if __name__ == "__main__":
    job_worker()
