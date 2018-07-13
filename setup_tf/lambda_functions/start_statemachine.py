import os
import boto3
from datetime import datetime 

sfn_arn= os.environ['SFN_ARN']
sfn = client('stepfunctions')
name = str(datetime.now()).split(' ')[1].split('.')[1]

def lambda_handler(event, context):
    try:
        response = sfn.start_execution(
            stateMachineArn=str(sfn_Arn),
            name=name,
            input='{}'
        )
    except Exception as e:
        print(e)
        print('Unable to start Step Function Execution.')
        raise(e)