import os
import boto3
from datetime import datetime
import botocore
import traceback

sfn_arn = os.environ['SFN_ARN']
model_prefix = os.environ['MODEL_PREFIX']
sfn = boto3.client('stepfunctions')
pipeline = boto3.client('codepipeline')

def put_job_success(job, message):
    """Notify CodePipeline of a successful job
    
    Args:
        job: The CodePipeline job ID
        message: A message to be logged relating to the job status
        
    Raises:
        Exception: Any exception thrown by .put_job_success_result()
    
    """
    print('Putting job success')
    print(message)
    pipeline.put_job_success_result(jobId=job)
  
def put_job_failure(job, message):
    """Notify CodePipeline of a failed job
    
    Args:
        job: The CodePipeline job ID
        message: A message to be logged relating to the job status
        
    Raises:
        Exception: Any exception thrown by .put_job_failure_result()
    
    """
    print('Putting job failure')
    print(message)
    pipeline.put_job_failure_result(jobId=job, failureDetails={'message': message, 'type': 'JobFailed'})

def lambda_handler(event, context):
    # Extract CodePipeline Job ID
    job_id = event['CodePipeline.job']['id']
    # Creating StateMachine based on job id
    name = model_prefix+'-'+str(job_id)
    try:
        sfn_response = sfn.start_execution(
            stateMachineArn=str(sfn_arn),
            name=name,
            input='{}'
        )
        print('Started Step Function Execution.')
        put_job_success(job_id, 'State Machine Started: '+sfn_response['executionArn'])
    except Exception as e:
        print(e)
        print('Unable to start Step Function Execution.')
        traceback.print_exec()
        put_job_failure(job_id, 'Function exception: '+str(e))
    return "Complete"