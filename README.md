# RoboStig - Workshop Setup

>__Based on the SageMaker Serverless Continuous Deployment repo. A CodePipeline has been added to automatically upload the initial training data as well as create the custom Keras Training and Hosting containers for the users and the instructor to demo.__

## Step 1: Workshop default SageMaker IAM Role.
This role will be used by SageMaker to assume access to the training and hosting instances as well as ECR and S3.
- Create IAM Role called `SageMaker-WS-Role`.
- Apply the following policy:
    {
    }
- Make note of the role name as it will be used in the CloudFormation template and the next step.

## Step 2: Workshop default IAM User.
This IAM user will be used to demonstrate the self-driving vehicle driving not he track. It will also be used as a backup if any users are unable to get a working solution.
- Create IAM User called `SageMaker-WS-User`.
- Apply the `SageMaker-WS-Role` (created in Step 1) to the user.
- Take note of the Access Key and Secret. This must be manually added to the presenter slides in __Model 4__.

## Step 3: Configure the GitHub Repository.
The repository will be the CodePipeline source for the training code and SageMaker containers.
- Fork the repository to your own GitHub account.
- Create a [GitHub Token](https://github.com/settings/tokens).

## Step 4: Deploy
At the end of the deployment, you should have the following:
1. S3 Bucket containing the source driving data for workshop users.
2. ECR repo containing the GPU and CPU custom SageMaker estimator for the workshop users to leverage in Module 4.
3. SageMaker inference endpoint for you to demonstrate the self-driving vehicle, as well as the workshop users to leverage if they are unable to complete any of the modules.

### Prepare Parameters
Update the [`parameters.json`](./cloudformation/parameters.json) with required information:
- __TBD__

### Deploy
- Execute the deployment:
```console
    $ cd cloudformation
    $ aws cloudformation ... --parameters file:///./parameters.json
```

### 