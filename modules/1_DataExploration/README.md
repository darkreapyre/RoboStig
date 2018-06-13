# Modeule 1: Data Exploration
## Overview
In this module you will explore the driving observations captured from the simulator. In order to create the most effective model for __RoboStig__ to use, exploring and understanding the data is crucial. Therefore, after exploring the data, you will determine the best features to extract and train on. Additionally, you will look at various options to transform and augment the data to ensure:

1. You have a sufficient amount of data for training.
2. The data is formatted correctly and "cleaned" for training.
3. The training samples provide a enough variation to ensure that the eventual model does not overfit the training samples.

To assist in these tasks, SageMaker provides hosted Jupyter notebooks that are pre-loaded with useful libraries for machine learning. To create a SageMaker notebook instance for this workshop, follow the instructions below.

## Creating a Notebook Instance

We'll start by creating a SageMaker notebook instance, which we will use for the other workshop modules.

1. Open the [SageMakerManagement Console](https://console.aws.amazon.com/sagemaker) in your browser and log into you AWS account.

2. In the upper-right corner of the AWS Management Console, confirm you are in the desired AWS region. Select __Oregon__.

3. To create a new notebook instance, go to **Notebook instances**, and click the **Create notebook instance** button at the top of the browser window.

![Notebook Instance](https://s3-us-west-2.amazonaws.com/robostig-assets-us-west-2/1/create.jpg)

4. Enter *RoboStig* into the **Notebook instance name** text box, and select *ml.t2.xlarge* for the **Notebook instance type**.

![Notebook Settings](https://s3-us-west-2.amazonaws.com/robostig-assets-us-west-2/1/settings.jpg)

5. For IAM role, choose **Create a new role**, and in the resulting pop-up modal, select **Any S3 bucket** under **S3 Buckets you specify – optional**. Click **Create role**.

![Bucket Access](https://s3-us-west-2.amazonaws.com/robostig-assets-us-west-2/1/bucket.jpg)

6. You will be taken back to the Create Notebook instance page.  Click **Create notebook instance**.

## Accessing the Notebook Instance

1. Wait for the server status to change to **InService**. This will take several minutes, possibly up to ten but likely less.

![Access Notebook](https://s3-us-west-2.amazonaws.com/robostig-assets-us-west-2/1/open.jpg)

2. Click **Open**. You will now see the Jupyter homepage for your notebook instance.

![Open Notebook](https://s3-us-west-2.amazonaws.com/robostig-assets-us-west-2/1/start.jpg)

## Starting the Module


1. In the upper-right corner of the Jupyter Notebook, click the **New** button and select **Terminal**.

![Terminal](https://s3-us-west-2.amazonaws.com/robostig-assets-us-west-2/1/terminal.jpg)

2. After the new terminal opens, download the workshop content by running the following commands:

```terminal
$ cd SageMaker/
$ git clone https://github.com/darkreapyre/RoboStig
$ exit
```

3. Closing the terminal tab will return you to the main Jupyter menu. Navigate to the `RoboStig\modules\1_DataExploratin\Data_Exploration.ipynb` Notebook to get get started exploring the driving observation data. Work through these steps on the Notebook to see explore the data:
- To run the notebook document step-by-step (one cell a time) by pressing shift + enter.
- To restart the kernel (i.e. the computational engine), click on the menu **Kernel** -> **Restart**. This can be useful to start over a computation from scratch (e.g. variables are deleted, open files are closed, etc…).
- More information on editing a notebook can be found on the [Notebook Basics](http://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb) page.