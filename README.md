# RoboStig - Workshop Setup

>__Based on the SageMaker Serverless Continuous Deployment [repo](https://github.com/aws-samples/serverless-sagemaker-orchestration). A CodePipeline has been added to automatically upload the initial training data as well as create the custom Keras Training and Hosting containers for the users and the instructor to demo.__

## Step 1: Workshop default SageMaker IAM Role.
This role will be used by SageMaker to assume access to the training and hosting instances as well as ECR and S3.
- Create IAM Role called `SageMaker-WS-Role` with the `AmazonSageMakerFullAccess` Policy.
- Make note of the role name as it will be used in the CloudFormation template and the next step.

## Step 2: Workshop default IAM User.
This IAM user will be used to demonstrate the self-driving vehicle driving not he track. It will also be used as a backup if any users are unable to get a working solution.
- Create IAM User called `SageMaker-WS-User`.
- Apply the `SageMaker-WS-Role` (created in Step 1) to the user.
- Take note of the Access Key and Secret. This must be manually added to the presenter slides in __Model 4__.

## Step 3: Create a default S3 bucket for the workshop.
The Bucket will contain the lambda assets for the Step Function as well as the Training Data source `.zip` file for __Module 1__ and extracted source for __Module 4__.
- Create the S3 Bucket called `sagemaker-workshop-<<AWS ACCOUNT ID>>-us-east-1` with the defaults and substitute `<<AWS ACCOUNT ID>>` with the account used to run the workshop.
- Apply the following Bucket Policy:
```json
    {
        "Id": "WSAccess",
        "Version": "2012-10-17",
        "Statement": [
            {
                "Action": [
                    "s3:Get*",
                    "s3:List*"
                ],
                "Effect": "Allow",
                "Resource": [
                    "arn:aws:s3:::sagemaker-workshop-<<AWS ACCOUNT ID>>-us-east-1",
                    "arn:aws:s3:::sagemaker-workshop-<<AWS ACCOUNT ID>>-us-east-1/*"
                ],
                "Principal": "*"
            }
        ]
    }
```

## Step 4: Configure the GitHub Repository.
The repository will be the CodePipeline source for the training code and SageMaker containers.
- Fork the repository to your own GitHub account.
- Create a [GitHub Token](https://github.com/settings/tokens).

## Step 5: Deploy
At the end of the deployment, you should have the following:
1. S3 Bucket containing the source driving data for workshop users.
2. ECR repo containing the GPU and CPU custom SageMaker estimator for the workshop users to leverage in Module 4.
3. SageMaker inference endpoint for you to demonstrate the self-driving vehicle, as well as the workshop users to leverage if they are unable to complete any of the modules.

### Create CloudFormation Package

```console
    $ cd cloudformation
    $ aws cloudformation package --region us-east-1 --s3-bucket sagemaker-workshop-<<AWS ACCOUNT ID>>-us-east-1 --template sagemaker_workshop_setup.yaml --output-template-file output.yaml
```

### Deploy CloudFormation Package

```console
    $ aws cloudformation deploy --region us-east-1 --template-file output.yaml --stack-name SageMaker-Workshop --capabilities CAPABILITY_NAMED_IAM --parameter-overrides GitHubUser=<<GitHub User>> GitHubRepo=<<GitHub Repository>> GitHubBranch=setup GitHubToken=<<GitHub Token>> SageMakerExecutionRole=<<SAGEMAKER ROLE ARN>> EmailAddress=<<UPDATE E-MAIL ADDRESS>> S3Bucket=sagemaker-workshop-<<AWS ACCOUNT ID>>-us-east-1
```
### Review CloudFormation Outputs
Review the `Outputs` Tab of for the CloudFormation Stack and update the course content with the following:
- `TrainingDataBucket`: This will be provided to the student in __Module 4__ as the location of the extracted training data should they wish to leverage the provide *Keras* Training and Hosting Containers.
- `ModelServingImage`: This will be provided to the student in __Module 4__ as the location of the hosting container should they wish to leverage the provide *Keras* Training and Hosting Containers.
- `TrainingDataDownloadURL`: This will be provided to the student in __Module 1__ as the location of the zipped training data to explore.
- `ModelTrainingImage`:  This will be provided to the student in __Module 4__ as the location of the GPU training container should they wish to leverage the provide *Keras* Training and Hosting Containers.

>__Note:__ You can follow the deployment of the workshop deployment by clicking on the value of the `CodePipelineURL` output.

## Step 6: Cleanup
Delete the following resources that aren't deleted by the CloudFormation template:
- Delete CodePipeline Artifacts and Workshop S3 Buckets.
- CloudWatch Logs.
    - `/aws/codebuild/*`
    - `/aws/lambda/SageMaker-Workshop*`
    - `/aws/sagemaker/Endpoints/pystig`
    - `/aws/sagemaker/TrainingJobs`
- `pystig` Container Repository.
- Delete SageMaker Endpoint, Endpoint Configuration and Model.
- IAM user `SageMaker-WS-User`.
- IAM role `SageMaker-WS-Role`.