import os
import click

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


@click.command()
@click.option("--message", "-m", required=True, help="Message to send to Slack")
def send_slack_message(message):
    slack_token = os.environ["SLACK_BOT_TOKEN"]
    client = WebClient(token=slack_token)
    user_id = os.environ["SLACK_USER_ID"]
    try:
        channel_id = client.conversations_open(users=user_id)["channel"]["id"]
        response = client.chat_postMessage(
            channel=channel_id,
            text=message,
        )
    except SlackApiError as e:
        print(f"Error posting message: {e}")


if __name__ == "__main__":
    send_slack_message()
