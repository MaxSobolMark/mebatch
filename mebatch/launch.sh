#!/bin/bash
# Command receives 5 arguments: name, ws_number, path, conda_env, and command
# Put them in variables
name=$1
ws_number = $2
path=$3
conda_env=$4
command=$5

kinit -l 1d -k -t ~/.keytab/maxsobolmark.keytab
tmux new-session -d -s ${name}
tmux new-window -t ${name} -n window

tmux send-keys -t ${name} 'cd ${path}' C-m
# Send slack message to notify that the command has started
/iris/u/maxsobolmark/decoupled_iql_env/bin/python /iris/u/maxsobolmark/mebatch/mebatch/slack.py --message 'Started running command on ws-${ws_number}: ${name}'
