from typing import Dict, List, Tuple, Optional
import click
import os
import getpass
import threading
import subprocess


VARIABLE_MARKER = "***"
ASKME_MARKER = "___ASKME___"
TIME_LIMIT_MARKER = "___TIME___"
MEMORY_MARKER = "___MEMORY___"
WS_USER = "maxsobolmark"


def get_current_conda_env() -> str:
    """
    Returns the name of the current conda environment.
    """
    print("Getting current conda environment...")
    result = subprocess.run(
        ["/iris/u/maxsobolmark/miniconda3/bin/conda", "info"], stdout=subprocess.PIPE
    )
    output = result.stdout.decode("utf-8")

    # Parse the output to get the active environment location
    env_lines = output.split("\n")
    active_env_line = [
        line for line in env_lines if line.strip().startswith("active environment")
    ]
    active_env_location = active_env_line[0].split(": ")[1]
    return active_env_location


def run_command_on_ws(
    command: str, ws_number: int, name: str, password: Optional[str] = None
) -> str:
    from fabric import Connection  # For running commands on workstations
    from paramiko.ssh_exception import AuthenticationException

    # Get password to connect to workstation
    password = password or getpass.getpass("Password: ")
    # Connect to workstation
    while True:
        try:
            connection = Connection(
                f"iris-ws-{ws_number}.stanford.edu",
                user=WS_USER,
                connect_kwargs={"password": password},
            )
            break
        except AuthenticationException:
            password = getpass.getpass("Incorrect password. Try again: ")
    print("Sending command to workstation...")

    def send_command(name, command):
        # Start a new tmux session
        name = name.replace(".", "_")
        connection.run(
            f"kinit {WS_USER} -l 3d -k -t ~/.keytab/maxsobolmark.keytab && aklog"
        )
        connection.run(f"tmux new-session -d -s {name}")
        # Create a new tmux window
        connection.run(f"tmux new-window -t {name} -n window")
        # Authenticate
        connection.run(
            f"tmux send-keys -t {name} 'kinit {WS_USER} -l 3d -k -t ~/.keytab/maxsobolmark.keytab &&"
            f" aklog' C-m"
        )
        # Cd to the right directory
        connection.run(f"tmux send-keys -t {name} 'cd {os.getcwd()}' C-m")
        # Send slack message to notify that the command has started
        connection.run(
            f"/iris/u/maxsobolmark/decoupled_iql_env/bin/python"
            f" /iris/u/maxsobolmark/mebatch/mebatch/slack.py --message 'Started running command on ws-{ws_number}: {name}'"
        )
        # Run the command
        connection.run(
            f"tmux send-keys -t {name}:window.0 'conda activate {get_current_conda_env()} &&"
            f" {command} && sleep 5 && tmux kill-session' C-m"
        )
        # # Send slack message to notify that the command has finished
        # connection.run(
        #     f"tmux send-keys -t {name}:window.0 '/iris/u/maxsobolmark/decoupled_iql_env/bin/python"
        #     f'/iris/u/maxsobolmark/mebatch/mebatch/slack.py --message "Finished running command on ws-{ws_number}: {name}"\' C-m'
        # )
        # # Set remain-on-exit to off so that the tmux session will close when the command finishes
        # connection.run(f"tmux set-option -t {name} remain-on-exit off")
        # connection.run(f"tmux send-keys -t {name}:window.0 'tmux kill-session' C-m")

    thread = threading.Thread(target=send_command, args=(name, command))
    thread.start()
    return password


def ebatch(
    name: str,
    command: str,
    ask_priority: bool,
    time_limit: int,
    memory: int,
    ws_password: Optional[str] = None,
) -> Tuple[bool, str, str]:
    if not ask_priority:
        priority = "h"
    else:
        priority = ""
        while (
            priority != "h"
            and priority != "l"
            and priority != "r"
            and priority != "s"
            and priority != "w"
        ):
            print(
                f"Sending {name}. (h)igh/(l)ow-priority/(r)un on this session/(s)kip/(w)orkstation?"
            )
            priority = input()
    if priority == "r":
        return True, priority, ws_password
    if priority == "s":
        return False, priority, ws_password
    if priority == "w":
        ws_number = ""
        while not ws_number.isdigit() or int(ws_number) > 18 or int(ws_number) < 1:
            print("Which workstation? (1-18)")
            ws_number = input()
        ws_password = run_command_on_ws(command, int(ws_number), name, ws_password)
        return False, priority, ws_password
    conf_file = "slconf" if priority == "l" else "slconf-hi"
    while not os.path.exists(f"{os.getcwd()}/{conf_file}"):
        print(f"Could not find {conf_file} in current directory.")
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
    return False, priority, None


def save_command_to_history(
    run_names: str,
    commands: str,
    previous_askme_responses: str,
    job_order: str,
    priority_responses: str,
):
    ask_order = job_order == list(range(len(job_order)))
    job_order_str = ""
    if ask_order:
        job_order_str = f"--ask-order {job_order}"
    with open("/iris/u/maxsobolmark/mebatch/mebatch_history.txt", "a") as f:
        f.write(
            f'\nmebatch {run_names} "{commands}" --previous_askme_responses {previous_askme_responses} {job_order_str}; priority responses: {"".join(priority_responses)} \n'
        )


def ask_job_order(run_names: List[str]) -> List[int]:
    """
    Asks the user for the order in which to run the commands.
    """
    print("Enter order in which to run commands:")
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
    python example.py --seed=$$$seed:0,1,2$$$
    --save_path=./exp/$$$seed$$$/$$$seed:first,second,third$$$
    --learning_rate=$$$lr:0.1,0.2,0.3$$$
    This example command will run ebatch 9=3*3 times (3 seeds time 3 learning rates).
    """
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
    ws_password = None
    for run_name, command in commands_to_run:
        run_on_this_session, priority, ws_password = ebatch(
            run_name, command, True, time_limit, memory, ws_password
        )
        if run_on_this_session:
            commands_to_run_on_this_session.append(command)
        priority_responses.append(priority)
        print("----------------------------------------")
    print("----------------------------------------")
    print(f"Running {len(commands_to_run_on_this_session)} commands on this session.")
    print("----------------------------------------")
    for i, command in enumerate(commands_to_run_on_this_session):
        print(
            f"Running command {i + 1}/{len(commands_to_run_on_this_session)}: {command}"
        )
        os.system(command)
        print("Command finished. Running kinitaklog")
        os.system(
            "kinit maxsobolmark -l 3d -k -t ~/.keytab/maxsobolmark.keytab && aklog"
        )
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
    variables_to_chosen_values: dictionary from variable name to tuples with
        choice index and chosen string.
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
    variable_name_end_position = variable_string.find(":")
    variable_name = (
        variable_string[:variable_name_end_position]
        if variable_name_end_position != -1
        else variable_string
    )
    if not (
        variable_name_end_position != -1 or variable_name in variables_to_chosen_values
    ):
        import pdb

        pdb.set_trace()
    assert (
        variable_name_end_position != -1 or variable_name in variables_to_chosen_values
    )
    if variable_name in variables_to_chosen_values:
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

    # variable hasn't been chosen, so choose every possible value
    commands_to_run = []
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
    if command.find(ASKME_MARKER) == -1:
        return command, ""
    askme_index = command.find(ASKME_MARKER)
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


# python example.py --seed=$$$seed:0,1,2$$$ --save_path=./exp/$$$seed$$$/$$$seed:first,second,third$$$ --learning_rate=$$$lr:0.1,0.2,0.3$$$
