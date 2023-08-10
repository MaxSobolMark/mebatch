from typing import Dict, Optional
import os
import click

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


def send_slack_message(message: str, thread_ts: Optional[str] = None) -> Dict[str, str]:
    slack_token = os.environ["SLACK_BOT_TOKEN"]
    client = WebClient(token=slack_token)
    user_id = os.environ["SLACK_USER_ID"]
    try:
        channel_id = client.conversations_open(users=user_id)["channel"]["id"]
        response = client.chat_postMessage(
            channel=channel_id,
            text=message,
            thread_ts=thread_ts,
        )
    except SlackApiError as e:
        print(f"Error posting message: {e}")

    return response


@click.command()
@click.option("--message", "-m", required=True, help="Message to send to Slack")
def send_slack_message_cli(message):
    send_slack_message(message)


if __name__ == "__main__":
    send_slack_message_cli()
