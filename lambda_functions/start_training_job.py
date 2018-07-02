import boto3
import os

# AWS provided containers for the Linear Learner model
CONTAINERS = {'us-west-2': '174872318107.dkr.ecr.us-west-2.amazonaws.com/linear-learner:latest',
              'us-east-1': '382416733822.dkr.ecr.us-east-1.amazonaws.com/linear-learner:latest',
              'us-east-2': '404615174143.dkr.ecr.us-east-2.amazonaws.com/linear-learner:latest',
              'eu-west-1': '438346466558.dkr.ecr.eu-west-1.amazonaws.com/linear-learner:latest'}
              
REGION = boto3.session.Session().region_name

# Number of features in data
FEATURE_DIM = os.environ['FEATURE_DIM']

# Instance type to train on
TRAINING_INSTANCE_TYPE = os.environ['TRAINING_INSTANCE_TYPE']

# Role to pass to SageMaker training job that has access to training data in S3, etc
SAGEMAKER_ROLE = os.environ['SAGEMAKER_ROLE']

sagemaker = boto3.client('sagemaker')


def lambda_handler(event, context):
    time = event['time']
    model_prefix = event['endpoint']
    train_manifest_uri = event['train_manifest_uri']
    container = CONTAINERS[REGION]
    s3_output_path = event['s3_output_path']
    name = '{}-{}'.format(model_prefix, time).replace(':', '-')
    print('Starting training job...')
    create_training_job(name, train_manifest_uri, container, s3_output_path)
    event['name'] = name
    event['container'] = container
    event['stage'] = 'Training'
    event['status'] = 'InProgress'
    event['message'] = 'Starting training job "{}"'.format(name)
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
                'feature_dim': FEATURE_DIM,
                'predictor_type': 'regressor',
                'mini_batch_size': '100'
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
                            'S3DataType': 'ManifestFile',
                            'S3Uri': train_manifest_uri,
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
                    'ContentType': 'text/csv',
                    'CompressionType': 'None'
                }
            ],
            OutputDataConfig={
                'S3OutputPath': s3_output_path
            },
            ResourceConfig={
                'InstanceType': TRAINING_INSTANCE_TYPE,
                'InstanceCount': 1,
                'VolumeSizeInGB': 50
            },
            StoppingCondition={
                'MaxRuntimeInSeconds': 86400
            }
        )
    except Exception as e:
        print(e)
        print('Unable to create training job.')
        raise(e)

"""
{'TrainingJobName': 'sagemaker-mxnet-2018-07-02-22-03-23-633',
 'TrainingJobArn': 'arn:aws:sagemaker:us-west-2:500842391574:training-job/sagemaker-mxnet-2018-07-02-22-03-23-633',
 'ModelArtifacts': {'S3ModelArtifacts': 's3://sagemaker-us-west-2-500842391574/sagemaker-mxnet-2018-07-02-22-03-23-633/output/model.tar.gz'},
 'TrainingJobStatus': 'Completed',
 'SecondaryStatus': 'Completed',
 'HyperParameters': {'batch_size': '64',
  'epochs': '12',
  'learning_rate': '0.001',
  'optmizer': '"adam"',
  'sagemaker_container_log_level': '20',
  'sagemaker_enable_cloudwatch_metrics': 'false',
  'sagemaker_job_name': '"sagemaker-mxnet-2018-07-02-22-03-23-633"',
  'sagemaker_program': '"model.py"',
  'sagemaker_region': '"us-west-2"',
  'sagemaker_submit_directory': '"s3://sagemaker-us-west-2-500842391574/sagemaker-mxnet-2018-07-02-22-03-23-633/source/sourcedir.tar.gz"'},
 'AlgorithmSpecification': {'TrainingImage': '520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet:1.1-gpu-py2',
  'TrainingInputMode': 'File'},
 'RoleArn': 'arn:aws:iam::500842391574:role/SageMaker',
 'InputDataConfig': [{'ChannelName': 'training',
   'DataSource': {'S3DataSource': {'S3DataType': 'S3Prefix',
     'S3Uri': 's3://sagemaker-us-west-2-500842391574/input_data',
     'S3DataDistributionType': 'FullyReplicated'}},
   'CompressionType': 'None',
   'RecordWrapperType': 'None'}],
 'OutputDataConfig': {'KmsKeyId': '',
  'S3OutputPath': 's3://sagemaker-us-west-2-500842391574'},
 'ResourceConfig': {'InstanceType': 'ml.p3.8xlarge',
  'InstanceCount': 1,
  'VolumeSizeInGB': 30},
 'StoppingCondition': {'MaxRuntimeInSeconds': 86400},
 'CreationTime': datetime.datetime(2018, 7, 2, 22, 3, 25, 531000, tzinfo=tzlocal()),
 'TrainingStartTime': datetime.datetime(2018, 7, 2, 22, 5, 9, 50000, tzinfo=tzlocal()),
 'TrainingEndTime': datetime.datetime(2018, 7, 2, 22, 8, 37, 204000, tzinfo=tzlocal()),
 'LastModifiedTime': datetime.datetime(2018, 7, 2, 22, 8, 37, 209000, tzinfo=tzlocal()),
 'ResponseMetadata': {'RequestId': '32dcace2-6b1f-461c-a384-855aa06099e6',
  'HTTPStatusCode': 200,
  'HTTPHeaders': {'content-type': 'application/x-amz-json-1.1',
   'date': 'Mon, 02 Jul 2018 22:53:32 GMT',
   'x-amzn-requestid': '32dcace2-6b1f-461c-a384-855aa06099e6',
   'content-length': '1796',
   'connection': 'keep-alive'},
  'RetryAttempts': 0}}



from sagemaker.mxnet.model import MXNetModel
sagemaker_model = MXNetModel(model_data = 's3://' + sagemaker_session.default_bucket() + '/model/model.tar.gz',
                                  role = role,
                                  entry_point = 'mnist.py')



  """