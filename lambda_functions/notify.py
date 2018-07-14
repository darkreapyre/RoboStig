import os
import boto3

SNS_TOPIC = os.environ['SNS']

sns = client('sns')

def lambda_handler(event, context):
    message = event['message']
    post_message(message)
    return event

def post_message(message):
    """ Posts message to SNS.
    Args:
        message (string): Message to post to SNS
    Returns:
        (None)
    """
    try:
        sns.publish(TargetArn=SNS_TOPIC, Message=message)
    except Exception as e:
        print(e)
        print('Unable to publish SNS message.')
        raise(e)