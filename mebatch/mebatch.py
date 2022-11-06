from typing import Dict, List, Tuple
import click
import os

VARIABLE_MARKER = "***"


def ebatch(name: str, command: str, ask_priority: bool):
    if not ask_priority:
        priority = "high"
    else:
        priority = ""
        while priority != "h" and priority != "l":
            print(f"Sending {name}. (h)igh/(l)ow-priority?")
            priority = input()
    conf_file = "slconf" if priority == "l" else "slconf-hi"
    assert os.path.exists(f"{os.getcwd()}/{conf_file}")
    with open(f"{os.getcwd()}/{conf_file}", "r") as config:
        config = config.read().replace("\n", " ")
        cmd = f"sbatch -J {name} {config} --wrap='{command}'"
        cmd = cmd.replace("$1", name)
        print(f"Running command: {cmd}")
        os.system(cmd)


@click.command()
@click.argument("run_names", required=True, type=str)
@click.argument("commands", required=True, type=str)
def mebatch(
    run_names: str,
    commands: str,
):
    """
    run_names and commands will contain encoded lists of options using the following format
    python example.py --seed=$$$seed:0,1,2$$$
    --save_path=./exp/$$$seed$$$/$$$seed:first,second,third$$$
    --learning_rate=$$$lr:0.1,0.2,0.3$$$
    This example command will run ebatch 9=3*3 times (3 seeds time 3 learning rates).
    """
    commands_to_run = mebatch_helper(run_names, commands, {})
    for run_name, command in commands_to_run:
        ebatch(run_name, command, True)


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


# python example.py --seed=$$$seed:0,1,2$$$ --save_path=./exp/$$$seed$$$/$$$seed:first,second,third$$$ --learning_rate=$$$lr:0.1,0.2,0.3$$$
