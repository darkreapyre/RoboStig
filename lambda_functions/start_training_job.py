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
    time_stamp = str(datetime.now()).split(' ')
    date = time_stamp[0]
    time = time_stamp[1].replace(':', '-').split('.')[0]
    ID = time_stamp[1].split('.')[1][-3:]
#    print('Start Time: {}'.format(time))
    model_prefix = os.environ['MODEL_PREFIX']
#    print('Model Prefix: {}'.format(model_prefix))
    train_manifest_uri = os.environ['S3_URI']
#    print('S3 URI: {}'.format(train_manifest_uri))
    container = CONTAINER
#    print('Using Container: {}'.format(container))
    s3_output_path = os.environ['OUTPUT_PATH']
#    print('Model Output Path: {}'.format(s3_output_path))
    name = '{}-{}-{}-{}'.format(model_prefix, date, time, ID)
    print('Starting {} training job ...'.format(name))
    create_training_job(name, train_manifest_uri, container, s3_output_path)
    event['name'] = name
    event['container'] = container
    event['stage'] = 'Training'
    event['status'] = 'InProgress'
    event['message'] = 'Started training job "{}"'.format(name)
    event['endpoint'] = model_prefix
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
                'batch_size': '16',
                'epochs': '25',
                'learning_rate': '0.0001',
                'gpu_count': '4'
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