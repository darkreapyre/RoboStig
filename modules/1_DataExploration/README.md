# Data Exploration
## Overview
In this module you will explore the driving observations captured from the simulator. In order to create the most effective model for __RoboStig__ to use, exploring and understanding the data is crucial. Therefore, after exploring the data, you will determine the best features to extract and train on. Additionally, you will look at various options to transform and augment the data to ensure:

1. You have a sufficient amount of data for training.
2. The data is formatted correctly and "cleaned" for training.
3. The training samples provide a enugh variation to ensure that the eventual model does not overfit the training samples.

To assist in these tasks, SageMaker provides hosted Jupyter notebooks that are pre-loaded with useful libraries for machine learning. To create a SageMaker notebook instance for this workshop, follow the instructions below.

## Creating a Notebook Instance

We'll start by creating an Amazon S3 bucket that will be used throughout the workshop.  We'll then create a SageMaker notebook instance, which we will use for the other workshop modules.

1. Open the [AWS Management Console](https://us-east-1.signin.aws.amazon.com/console) in your browser and log into you AWS account. 

2. In the upper-right corner of the AWS Management Console, confirm you are in the desired AWS region. Select N. Virginia, Oregon, Ohio, or Ireland.

3. Click on Amazon SageMaker from the list of all services.  This will bring you to the Amazon SageMaker console homepage.

![Services in Console](./images/console-services.png)

4. To create a new notebook instance, go to **Notebook instances**, and click the **Create notebook instance** button at the top of the browser window.

![Notebook Instances](./images/notebook-instances.png)

5. Type smworkshop-[First Name]-[Last Name] into the **Notebook instance name** text box, and select ml.m4.xlarge for the **Notebook instance type**.

![Create Notebook Instance](./images/notebook-settings.png)

6. For IAM role, choose **Create a new role**, and in the resulting pop-up modal, select **Specific S3 buckets** under **S3 Buckets you specify – optional**. In the text field, paste the name of the S3 bucket you created above, AND the following bucket name separated from the first by a comma:  `gdelt-open-data`.  The combined field entry should look similar to ```smworkshop-john-smith, gdelt-open-data```. Click **Create role**.

![Create IAM role](./images/role-popup.png)

7. You will be taken back to the Create Notebook instance page.  Click **Create notebook instance**.

## Access theNotebook Instance

1. Wait for the server status to change to **InService**. This will take several minutes, possibly up to ten but likely less.

![Access Notebook](./images/open-notebook.png)

2. Click **Open**. You will now see the Jupyter homepage for your notebook instance.

![Open Notebook](./images/jupyter-homepage.png)








__Create Notebook Instance: `ml.t2.xlarge`__