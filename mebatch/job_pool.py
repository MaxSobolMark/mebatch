"""The job pool gets started by mebatch, and runs jobs in the background.
The job pool gets assigned an id, and it reads the job queue associated with that id until the queue
is empty. When the queue is empty, and all jobs have finished, the job pool exits.

Before the job pool is created, directory {mebatch_dir}/job_pools/active_pools/{id} is created,
containing a file called "new_jobs.txt", where each line is "{job_name},{command}".
When the file is not empty, the job pool reads the first line, and starts a job with that command.
This creates a file called "{job_name}_incomplete.txt" in the same directory with the command.
Once the job is finished, the file is renamed to "{job_name}_complete.txt".
 """

from typing import List
import click
import signal  # to handle slurm terminations, e.g. preemptions.
import os
import time
import concurrent.futures  # For the process pool.
import subprocess
from filelock import FileLock  # To lock the job queue file.
from mebatch.slack import send_slack_message

SAVE_PATH = os.environ.get("MEBATCH_PATH", f'/iris/u/{os.environ["USER"]}/mebatch')


class GracefulKiller:
    """This class is used to handle signals from the OS, e.g. SIGINT, SIGTERM."""

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
    pipe_stdout: bool = True,
    pipe_stderr: bool = True,
    slack_thread: str = None,
):
    """Runs the job synchronously. To be called by the job pool."""
    # Create the incomplete file.
    incomplete_file_path = os.path.join(
        SAVE_PATH, "job_pools", "active_pools", id, f"{job_name}_incomplete.txt"
    )
    stdout_file_path = os.path.join(
        SAVE_PATH, "job_pools", "active_pools", id, f"{job_name}_stdout.txt"
    )
    stderr_file_path = os.path.join(
        SAVE_PATH, "job_pools", "active_pools", id, f"{job_name}_stderr.txt"
    )
    with open(incomplete_file_path, "w") as f:
        f.write(command)
    # Send a slack message alerting that the job started.
    if slack_thread:
        send_slack_message(
            f"Job {job_name} started.",
            thread_ts=slack_thread,
        )
    # Run the job.
    if pipe_stdout:
        stdout_file = open(stdout_file_path, "wb", encoding="utf-8")
    else:
        stdout_file = None
    if pipe_stderr:
        stderr_file = open(stderr_file_path, "wb", encoding="utf-8")
    else:
        stderr_file = None
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=stdout_file if pipe_stdout else None,
        stderr=stderr_file if pipe_stderr else None,
    )
    # Wait for the job to finish.
    error_code = process.wait()
    if pipe_stdout:
        stdout_file.close()
    if pipe_stderr:
        stderr_file.close()
    # Rename the incomplete file to complete.
    complete_file_path = os.path.join(
        SAVE_PATH, "job_pools", "active_pools", id, f"{job_name}_complete.txt"
    )
    os.rename(incomplete_file_path, complete_file_path)
    # Send a slack message alerting that the job finished.
    if slack_thread:
        if error_code == 0:
            send_slack_message(
                f"🟢 Job {job_name} finished successfully.",
                thread_ts=slack_thread,
            )
        else:
            send_slack_message(
                f"🔴 Job {job_name} finished with error code {error_code}.",
                thread_ts=slack_thread,
            )


def run_new_jobs(
    executor: MonitoredProcessPoolExecutor,
    new_jobs_file_path: str,
    new_jobs_file_lock: FileLock,
    pipe_stdout: bool = True,
    pipe_stderr: bool = True,
    slack_thread: str = None,
) -> List[concurrent.futures.Future]:
    """Run new jobs."""
    futures = []
    with new_jobs_file_lock:
        with open(new_jobs_file_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            job_name, command = line.split(",")
            command = command.strip()
            futures.append(
                executor.submit(
                    run_one_job,
                    job_name,
                    command,
                    pipe_stdout=pipe_stdout,
                    pipe_stderr=pipe_stderr,
                    slack_thread=slack_thread,
                )
            )
        # Delete the lines from the file.
        with open(new_jobs_file_path, "w") as f:
            f.write("")
    return futures


@click.command()
@click.argument("id", type=str)
@click.argument("num_parallel_jobs", type=int)
@click.option(
    "--pipe-stdout/--no-pipe-stdout",
    default=True,
    help="Whether to pipe stdout to a file.",
)
@click.option(
    "--pipe-stderr/--no-pipe-stderr",
    default=True,
    help="Whether to pipe stderr to a file.",
)
@click.option(
    "--send-slack-messages/--no-send-slack-messages",
    default=True,
    help="Whether to send slack messages.",
)
def job_pool(
    id: str,
    num_parallel_jobs: int,
    pipe_stdout: bool = True,
    pipe_stderr: bool = True,
    send_slack_messages: bool = True,
):
    assert num_parallel_jobs > 0, "num_parallel_jobs must be positive."
    new_jobs_file_path = f"{SAVE_PATH}/job_pools/active_pools/{id}/new_jobs.txt"
    new_jobs_file_lock_path = (
        f"{SAVE_PATH}/job_pools/active_pools/{id}/new_jobs.txt.lock"
    )
    assert os.path.exists(
        new_jobs_file_path
    ), f"File new_jobs_file_path does not exist."
    new_jobs_file_lock = FileLock(new_jobs_file_lock_path, timeout=5)

    executor = MonitoredProcessPoolExecutor(
        max_workers=num_parallel_jobs,
    )
    killer = GracefulKiller()
    slack_thread = None
    # Send initial slack message.
    if send_slack_messages:
        response = send_slack_message(
            f"Job pool {id} started with {num_parallel_jobs} parallel jobs."
        )
        if not response["ok"]:
            print("Slack message failed to send.")
        else:
            slack_thread = response["ts"]

    # Send to run the jobs in new_jobs.txt.
    run_new_jobs(
        executor=executor,
        new_jobs_file_path=new_jobs_file_path,
        new_jobs_file_lock=new_jobs_file_lock,
        pipe_stdout=pipe_stdout,
        pipe_stderr=pipe_stderr,
    )
    # Keep running until all jobs are finished.
    while not killer.kill_now and executor.get_pool_usage() > 0:
        # Sleep for 2 seconds before checking again to avoid using resources.
        time.sleep(2)
        # Check if there are any new jobs.
        run_new_jobs(
            executor=executor,
            new_jobs_file_path=new_jobs_file_path,
            new_jobs_file_lock=new_jobs_file_lock,
            pipe_stdout=pipe_stdout,
            pipe_stderr=pipe_stderr,
            slack_thread=slack_thread,
        )
    if killer.kill_now:
        print("Job pool killed.")
        if send_slack_messages:
            send_slack_message(
                f"Job pool {id} was killed ☠️.",
                thread_ts=slack_thread,
            )
        # Rename the pool directory to killed.
        os.rename(
            f"{SAVE_PATH}/job_pools/active_pools/{id}",
            f"{SAVE_PATH}/job_pools/killed_pools/{id}",
        )
    else:
        print("Job pool finished.")
        if send_slack_messages:
            send_slack_message(
                f"Job pool {id} finished successfully 📈.",
                thread_ts=slack_thread,
            )
        # Rename the pool directory to finished.
        os.rename(
            f"{SAVE_PATH}/job_pools/active_pools/{id}",
            f"{SAVE_PATH}/job_pools/finished_pools/{id}",
        )


def get_active_pools() -> List[str]:
    """Get the list of active pools."""
    return os.listdir(f"{SAVE_PATH}/job_pools/active_pools")


def get_finished_pools() -> List[str]:
    """Get the list of finished pools."""
    return os.listdir(f"{SAVE_PATH}/job_pools/finished_pools")


def get_killed_pools() -> List[str]:
    """Get the list of killed pools."""
    return os.listdir(f"{SAVE_PATH}/job_pools/killed_pools")


def make_pool(id: str):
    """Make a new pool, setting up the directory and new_jobs file."""
    # Make the directory.
    os.mkdir(f"{SAVE_PATH}/job_pools/active_pools/{id}")
    # Make the new_jobs file.
    with open(f"{SAVE_PATH}/job_pools/active_pools/{id}/new_jobs.txt", "w") as f:
        f.write("")


def add_job_to_pool(job_pool_id: str, job_name: str, command: str):
    """Add a job to the pool."""
    # Add the job to the new_jobs file.
    new_jobs_lock = FileLock(
        f"{SAVE_PATH}/job_pools/active_pools/{job_pool_id}/new_jobs.txt.lock", timeout=5
    )
    with new_jobs_lock:
        with open(
            f"{SAVE_PATH}/job_pools/active_pools/{job_pool_id}/new_jobs.txt", "a"
        ) as f:
            f.write(f"{job_name},{command}\n")


if __name__ == "__main__":
    job_pool()
