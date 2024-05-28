from typing import Dict, List, Tuple, Optional
import click
import os
import tensorflow as tf  # For GCS support.
from filelock import FileLock  # To lock the job queue file.
from mebatch.GCS_file_lock import GCSFileLock
from mebatch.job_pool import get_active_pools, make_pool, add_job_to_pool
from mebatch.job_worker import read_last_online_times


VARIABLE_MARKER = "***"
ASKME_MARKER = "___ASKME___"
TIME_LIMIT_MARKER = "___TIME___"
MEMORY_MARKER = "___MEMORY___"
SAVE_PATH = os.environ.get("MEBATCH_PATH", f'/iris/u/{os.environ["USER"]}/mebatch')
# The command to run after each run is finished (only if the job is run on this session).
# This is useful for e.g. running kinit aklog with a keytab file on workstations.
COMMAND_AFTER_RUNS = os.environ.get("MEBATCH_COMMAND_AFTER_RUNS", "")
JOB_POOL_PIPE_STDOUT = (
    os.environ.get("MEBATCH_JOB_POOL_PIPE_STDOUT", "true").lower() == "true"
)
JOB_POOL_PIPE_STDERR = (
    os.environ.get("MEBATCH_JOB_POOL_PIPE_STDERR", "true").lower() == "true"
)
JOB_POOL_SEND_SLACK_MESSAGES = (
    os.environ.get("MEBATCH_JOB_POOL_SEND_SLACK_MESSAGES", "true").lower() == "true"
)


def ebatch(
    name: str,
    command: str,
    time_limit: int,
    memory: int,
    priority: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Sends a command to the SLURM queue.

    Args:
        name: The name of the job.
        command: The command to run.
        ask_priority: Whether to ask the user for the priority of the job. If False, defaults to
            high.
        time_limit: The time limit for the job in hours.
        memory: The memory limit for the job in GB.

    Returns:
        A tuple containing:
            - Whether the job is to be run on this session.
            - The priority of the job.
    """
    if priority is None:
        priority = ""
        while priority not in ["h", "l", "r", "s", "S", "p", "w"]:
            print(
                f"Sending {name}. (h)igh/(l)ow-priority/(r)un on this session/(s)kip/job (p)ool/(w)orker?"
            )
            priority = input()
    if priority == "r":
        return True, priority
    if priority.lower() == "s":
        return False, priority
    if priority == "p":
        active_pools = get_active_pools()
        if len(active_pools) == 0:
            print("No active pools found.")
        else:
            print("Active pools:")
            for pool_id in active_pools:
                print(f"- {pool_id}")
        print("Which pool to add to? (Enter a new name to create a new pool.)")
        pool_id = input()
        if pool_id not in active_pools:
            print("Creating new pool. High or low priority? (h/l)")
            priority = input().lower()
            while priority not in ["h", "l"]:
                print("Please enter h or l.")
                priority = input().lower()
            print("How many parallel jobs? (Enter a number or leave blank for 2.)")
            num_parallel_jobs = input()
            while not num_parallel_jobs.isdigit():
                if num_parallel_jobs == "":
                    num_parallel_jobs = "2"
                else:
                    print("Please enter a number.")
                    num_parallel_jobs = input()
            make_pool(pool_id)
            name = f"MEBatch-JobPool-{pool_id}-x{num_parallel_jobs}{priority.upper()}"
            command = f"python -m mebatch.job_pool --id={pool_id} --num_parallel_jobs={num_parallel_jobs}"
            if not JOB_POOL_PIPE_STDOUT:
                command += " --no-pipe-stdout"
            if not JOB_POOL_PIPE_STDERR:
                command += " --no-pipe-stderr"
            if not JOB_POOL_SEND_SLACK_MESSAGES:
                command += " --no-send-slack-messages"
        else:
            print("Adding to existing pool...")
        add_job_to_pool(pool_id, name, command)
        if pool_id in active_pools:
            return False, priority
    if priority == "w":
        while not os.path.exists(f"{SAVE_PATH}/workers.txt"):
            print(f"Could not find workers.txt in mebatch path ({SAVE_PATH}).")
            print("Go ahead and create it, I'll wait. (Press 'Enter' when done.)")
            input()
        # Print available workers
        with open(f"{SAVE_PATH}/workers.txt", "r") as workers:
            workers = workers.read().splitlines()
        worker_id_to_mebatch_dir = {
            worker.split()[0]: worker.split()[1] for worker in workers
        }
        worker_id_to_is_tpu = {
            worker.split()[0]: worker.split()[2] for worker in workers
        }
        worker_id_to_is_tpu = {
            id: is_tpu == "tpu" for id, is_tpu in worker_id_to_is_tpu.items()
        }
        # print(f"Available workers: {', '.join(worker_id_to_mebatch_dir.keys())}")
        worker_id_to_last_online_time = read_last_online_times(
            workers[0].split()[1], worker_id_to_mebatch_dir.keys()
        )
        print("Available workers:")
        for worker_id, last_online_time in worker_id_to_last_online_time.items():
            print(f"{worker_id} (last online: {last_online_time})")

        print("Which worker to run on?")
        worker_id = input()
        while worker_id not in worker_id_to_mebatch_dir:
            print("Please enter a valid worker ID.")
            worker_id = input()
        worker_mebatch_dir = worker_id_to_mebatch_dir[worker_id]
        worker_is_tpu = worker_id_to_is_tpu[worker_id]
        if worker_is_tpu:
            print("How many TPU cores to use? (either 1, 2, or 4)")
            num_tpu_cores = input()
            while num_tpu_cores not in ["1", "2", "4"]:
                print("Please enter 1, 2, or 4.")
                num_tpu_cores = input()
        new_jobs_file_lock_path = (
            f"{worker_mebatch_dir}/job_pools/active_pools/{worker_id}/new_jobs.txt.lock"
        )
        new_jobs_file_lock = (
            FileLock(new_jobs_file_lock_path)
            if "gs://" not in worker_mebatch_dir
            else GCSFileLock(new_jobs_file_lock_path)
        )
        with new_jobs_file_lock:
            with tf.io.gfile.GFile(
                tf.io.gfile.join(
                    worker_mebatch_dir,
                    "job_pools/active_pools",
                    worker_id,
                    "new_jobs.txt",
                ),
                "a",
            ) as new_jobs_file:
                if worker_is_tpu:
                    new_jobs_file.write(f"{name}\t{command}\t{num_tpu_cores}\n")
                else:
                    new_jobs_file.write(f"{name}\t{command}\n")
        return False, priority

    conf_file = "slconf" if priority == "l" else "slconf-hi"
    while not os.path.exists(f"{os.getcwd()}/{conf_file}"):
        print(
            f"Could not find {conf_file} in current directory. Example files can be found at /iris/u/maxsobolmark/slconf-(hi-)example"
        )
        print('Go ahead and copy it, I\'ll wait. (Press "Enter" when done.)')
        input()
    with open(f"{os.getcwd()}/{conf_file}", "r") as config:
        config = config.read().replace("\n", " ")
        cmd = f"sbatch -J {name} {config} --wrap='{command}'"
        cmd = cmd.replace("$1", name)
        cmd = cmd.replace(TIME_LIMIT_MARKER, f"{time_limit}:00:00")
        cmd = cmd.replace(MEMORY_MARKER, f"{memory}G")
        print(f"Running command: {cmd}")
        os.system(cmd)
    return False, priority


def save_command_to_history(
    run_names: str,
    commands: str,
    previous_askme_responses: str,
    job_order: str,
    priority_responses: str,
):
    """
    Saves the command to the history file.
    """
    ask_order = job_order == list(range(len(job_order)))
    job_order_str = ""
    if ask_order:
        job_order_str = f"--ask-order {job_order}"
    with open(f"{SAVE_PATH}/mebatch_history.txt", "a") as f:
        f.write(
            f'\nmebatch {run_names} "{commands}" --previous_askme_responses {previous_askme_responses} {job_order_str}; priority responses: {"".join(priority_responses)} \n'
        )


def ask_job_order(run_names: List[str]) -> List[int]:
    """
    Asks the user for the order in which to run the commands.
    Expects the user to enter a comma-separated list of integers.
    """
    print("Enter order in which to run commands, comma-separated:")
    for i, name in enumerate(run_names):
        print(f"{i}: {name}")
    order = input()
    order = [int(i) for i in order.split(",")]
    return order


@click.command()
@click.argument("run_names", required=True, type=str)
@click.argument("commands", required=True, type=str)
@click.option("--previous_askme_responses", default="", type=str)
@click.option(
    "--ask_order",
    is_flag=True,
    default=False,
    help="Ask for job priority before sending jobs",
)
@click.option(
    "--time_limit",
    default=130,
    help="Time limit for each job in hours",
    type=int,
)
@click.option(
    "--memory",
    default=32,
    help="Memory limit for each job in GB",
    type=int,
)
def mebatch(
    run_names: str,
    commands: str,
    previous_askme_responses: str = "",
    ask_order: bool = False,
    time_limit: int = 130,
    memory: int = 20,
):
    """
    run_names and commands will contain encoded lists of options using the following format
    python example.py --seed=***seed:0,1,2***
    --save_path=./exp/***seed***/***seed:first,second,third***
    --learning_rate=***lr:0.1,0.2,0.3***
    This example command will run ebatch 9=3*3 times (3 seeds * 3 learning rates).
    """
    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        os.makedirs(f"{SAVE_PATH}/job_pools")
        os.makedirs(f"{SAVE_PATH}/job_pools/active_pools")
        os.makedirs(f"{SAVE_PATH}/job_pools/finished_pools")
        os.makedirs(f"{SAVE_PATH}/job_pools/killed_pools")
    commands_to_run = mebatch_helper(run_names, commands, {})
    askme_commands_to_run = []
    previous_askme_responses = previous_askme_responses.split(",")
    askme_responses = []
    priority_responses = []
    for run_name, command in commands_to_run:
        command, new_askme_responses = askme_helper(command, previous_askme_responses)
        askme_commands_to_run.append((run_name, command))
        askme_responses.extend(new_askme_responses)
    commands_to_run = askme_commands_to_run
    commands_to_run_on_this_session = []
    if ask_order:
        order = ask_job_order([name for name, _ in commands_to_run])
        assert len(order) == len(commands_to_run) and set(order) == set(
            range(len(order))
        )

        commands_to_run = [commands_to_run[i] for i in order]
    else:
        order = list(range(len(commands_to_run)))
    print("----------------------------------------")
    print(f"Running {len(commands_to_run)} commands.")
    print("----------------------------------------")
    for run_name, command in commands_to_run:
        run_on_this_session, priority = ebatch(run_name, command, time_limit, memory)
        if run_on_this_session:
            commands_to_run_on_this_session.append(command)
        priority_responses.append(priority)
        if priority == "S":
            print("Skipping rest of commands due to priority 'S'")
            break
        print("----------------------------------------")
    print("----------------------------------------")
    print(f"Running {len(commands_to_run_on_this_session)} commands on this session.")
    print("----------------------------------------")
    for i, command in enumerate(commands_to_run_on_this_session):
        print(
            f"Running command {i + 1}/{len(commands_to_run_on_this_session)}: {command}"
        )
        os.system(command)
        print("Command finished.")
        if COMMAND_AFTER_RUNS:
            print(f"Running '{COMMAND_AFTER_RUNS}'")
            os.system(COMMAND_AFTER_RUNS)
        print("----------------------------------------")
    if len(askme_responses) > 0:
        print("To use askme responses for next run, add:")
        print(f"--previous_askme_responses={','.join(askme_responses)}")
    order = ",".join([str(i) for i in order])
    save_command_to_history(
        run_names, commands, previous_askme_responses, order, priority_responses
    )


def mebatch_helper(
    run_names: str,
    commands: str,
    variables_to_chosen_values: Dict[str, Tuple[int, str]] = {},
) -> List[Tuple[str, str]]:
    """
    Replaces all variables in run_names and commands with the chosen values, and returns
    all combinations of run_names and commands.

    Args:
        run_names: string with run names, with variables marked with VARIABLE_MARKER.
        commands: string with commands, with variables marked with VARIABLE_MARKER.
        variables_to_chosen_values: dictionary from variable name to tuples with
            choice index and chosen string.

    Returns:
        List of tuples with run_names and commands.
    """

    if run_names.find(VARIABLE_MARKER) == -1 and commands.find(VARIABLE_MARKER) == -1:
        # No more variables to replace, we can return the commands as they are
        return [(run_names, commands)]

    # Assign the first variable, and make recursive call
    first_unassigned_variable_index = run_names.find(VARIABLE_MARKER)
    unassigned_variable_in_run_names = first_unassigned_variable_index != -1
    string_with_unassigned_variable = (
        run_names if unassigned_variable_in_run_names else commands
    )
    if not unassigned_variable_in_run_names:
        first_unassigned_variable_index = commands.find(VARIABLE_MARKER)
    variable_string_end = string_with_unassigned_variable.find(
        VARIABLE_MARKER, first_unassigned_variable_index + len(VARIABLE_MARKER)
    )
    variable_string = string_with_unassigned_variable[
        first_unassigned_variable_index + len(VARIABLE_MARKER) : variable_string_end
    ]
    # variable_string contains only the variable info, without the marker.
    # E.g. "seed:0,1,2" or "lr:0.1,0.2,0.3", or "seed"
    variable_name_end_position = variable_string.find(":")
    variable_name = (
        variable_string[:variable_name_end_position]
        if variable_name_end_position != -1
        else variable_string
    )
    if not (
        variable_name_end_position != -1 or variable_name in variables_to_chosen_values
    ):
        print(f"Variable {variable_name} was referenced but not assigned.")
        import pdb

        pdb.set_trace()

    if variable_name in variables_to_chosen_values:
        # Variable was already assigned, so we can replace it with the chosen value and
        # make recursive call.
        variable_value = variables_to_chosen_values[variable_name][1]
        if variable_name_end_position != -1:
            # User passed additional variable values after ':', so replace value for that.
            value_options = variable_string[variable_name_end_position + 1 :].split(",")
            variable_value = value_options[variables_to_chosen_values[variable_name][0]]
        truncated_string = (
            string_with_unassigned_variable[:first_unassigned_variable_index]
            + str(variable_value)
            + string_with_unassigned_variable[
                first_unassigned_variable_index
                + len(VARIABLE_MARKER) * 2
                + len(variable_string) :
            ]
        )
        if unassigned_variable_in_run_names:
            return mebatch_helper(
                truncated_string,
                commands,
                variables_to_chosen_values,
            )
        else:
            return mebatch_helper(
                run_names, truncated_string, variables_to_chosen_values
            )

    # Variable hasn't been chosen, so choose every possible value
    commands_to_run = []
    # value_options is what comes after the ":" in the variable string, e.g. [0,1,2]
    value_options = variable_string[variable_name_end_position + 1 :].split(",")
    for i, variable_value in enumerate(value_options):
        truncated_string = (
            string_with_unassigned_variable[:first_unassigned_variable_index]
            + str(variable_value)
            + string_with_unassigned_variable[
                first_unassigned_variable_index
                + len(VARIABLE_MARKER) * 2
                + len(variable_string) :
            ]
        )
        if unassigned_variable_in_run_names:
            commands_to_run += mebatch_helper(
                truncated_string,
                commands,
                {**variables_to_chosen_values, **{variable_name: (i, variable_value)}},
            )
        else:
            commands_to_run += mebatch_helper(
                run_names,
                truncated_string,
                {**variables_to_chosen_values, **{variable_name: (i, variable_value)}},
            )
    return commands_to_run


def askme_helper(
    command: str, previous_responses: Optional[List[str]]
) -> Tuple[str, List[str]]:
    """
    Gives the user the option to manually specify parts of the command.
    """
    askme_index = command.find(ASKME_MARKER)
    if askme_index == -1:
        return command, ""
    responses = []
    while askme_index != -1:
        print("previous_responses", previous_responses)
        if len(previous_responses) > 0 and previous_responses[0] != "":
            response = previous_responses.pop(0)
        else:
            print(f'Completing command "{command}"')
            print("Enter value for first askme marker:")
            response = input()
            responses.append(response)
        command = (
            command[:askme_index]
            + response
            + command[askme_index + len(ASKME_MARKER) :]
        )
        askme_index = command.find(ASKME_MARKER)
    return command, responses
