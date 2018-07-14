from datetime import datetime
import boto3
import os

# Instance type to train on
TRAINING_INSTANCE_TYPE = os.environ['TRAINING_INSTANCE_TYPE']

# Role to pass to SageMaker training job that has access to training data in S3, etc
SAGEMAKER_ROLE = os.environ['SAGEMAKER_ROLE']

# Get the container and tag for the training job
CONTAINER = str(os.environ['CONTAINER']) + ':' + str(os.environ['TAG'])

sagemaker = boto3.client('sagemaker')


def lambda_handler(event, context):
    time = str(datetime.now()).split(' ')[0]
    model_prefix = os.environ['MODEL_PREFIX']
    train_manifest_uri = os.environ['S3_URI']
    container = CONTAINER
    s3_output_path = os.environ['OUTPUT_PATH']
    name = '{}-{}'.format(model_prefix, time)
    print('Starting training job ...')
    create_training_job(name, train_manifest_uri, container, s3_output_path)
    event['name'] = name
    event['container'] = container
    event['stage'] = 'Training'
    event['status'] = 'InProgress'
    event['message'] = 'Started training job "{}"'.format(name)
    return event


def create_training_job(name, train_manifest_uri, container, s3_output_path):
    """ Start SageMaker training job
    Args:
        name (string): Name to label training job with
        train_manifest_uri (string): URI to training data manifest file in S3
        container (string): Registry path of the Docker image that contains the training algorithm
        s3_output_path (string): Path of where in S3 bucket to output model artifacts after training
    Returns:
        (None)
    """
    try:
        response = sagemaker.create_training_job(
            TrainingJobName=name,
            HyperParameters={
                'batch_size': '32',
                'epochs': '12',
                'learning_rate': '0.0001'
            },
            AlgorithmSpecification={
                'TrainingImage': container,
                'TrainingInputMode': 'File'
            },
            RoleArn=SAGEMAKER_ROLE,
            InputDataConfig=[
                {
                    'ChannelName': 'train',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': train_manifest_uri,
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
                    'RecordWrapperType': 'None',
                    'CompressionType': 'None'
                }
            ],
            OutputDataConfig={
                'S3OutputPath': s3_output_path
            },
            ResourceConfig={
                'InstanceType': TRAINING_INSTANCE_TYPE,
                'InstanceCount': 1,
                'VolumeSizeInGB': 30
            },
            StoppingCondition={
                'MaxRuntimeInSeconds': 86400
            }
        )
    except Exception as e:
        print(e)
        print('Unable to create training job.')
        raise(e)