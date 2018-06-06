# RoboStig - Leveraging Amazon SageMaker to build an Autonomous Self-driving Car

## Overview

If you're familiar with the British motoring television series [Top Gear](https://en.wikipedia.org/wiki/Top_Gear_(2002_TV_series)), then you are very familiar with [The Stig](https://www.topgear.com/car-news/stig).

>__"The Stig is a character on the British motoring television show Top Gear. The character is a play on the anonymity of racing drivers' full-face helmets, with the running joke that nobody knows who is inside the Stig's racing suit. The Stig's primary role is setting lap times for cars tested on the show." - Wikipedia__

In this workshop, you will create your own __RoboStig__  by way of a Machine Learning (ML) technique called, __Behavioral Cloning__. This is where an ML model learns to mimmic procedural knowledge through observation. In essence you will teach your __RoboStig__ to test drive a virtual car around a track by watching you drive the vehicle.

In addition to using a [simulator](https://github.com/udacity/self-driving-car-sim) - developed by [Udacity](https://www.udacity.com/) - that captures the observation data, you will also leverage [Amazon SageMaker](https://aws.amazon.com/sagemaker) and construct a *Machine Learning Pipeline* to do the following:

1. Explore and augment the training data.
2. Train and optimize your __RoboStig__ model using SageMaker's built-in [image classification algorithm](https://docs.aws.amazon.com/sagemaker/latest/dg/image-classification.html).
3. Train and optimize your __RoboStig__ model with your own algorithm based on the __End-to-End Deep Learning for Self-Driving Cars__ paper from [NVIDIA](https://devblogs.nvidia.com/deep-learning-self-driving-cars/).
4. Evaluate the performance of the model variants.
5. Use the best model for __RoboStig__ to drive the vehicle autonomously around the track in the simulator, thus cloning you behavior.

---

## Prerequisites

### AWS Account

In order to complete this workshop you'll need an AWS Account with access to create AWS IAM, S3 and SageMaker resources. The code and instructions in this workshop assume only one student is using a given AWS account at a time. If you try sharing an account with another student, you'll run into naming conflicts for certain resources. You can work around these by appending a unique suffix to the resources that fail to create due to conflicts, but the instructions do not provide details on the changes required to make this work.

<!--All of the resources you will launch as part of this workshop are eligible for the AWS free tier if your account is less than 12 months old. See the [AWS Free Tier page](https://aws.amazon.com/free/) for more details.-->

### Docker

In order to test the final __RoboStig__ model, you will need to install Docker for operating system and version. Follow the *Installation Instructions* for your OS in the table below:

| OS                                       | Installation<br>Instruction               | Docker System               | Shell                      | Access Jupyter at |
|:-----------------------------------------|:-----------------------------------------:|:----------------------------|:--------------------------:|:-----------------:|
| Linux                                    | [Here](https://docs.docker.com/engine/installation/linux/)           | Docker for Linux            | `bash`                     | `localhost:8888`  |
| MacOS <br>>= 10.10.3 (Yosemite)              | [Here](https://docs.docker.com/docker-for-mac/)             | Docker for Mac              | `bash`                     | `localhost:8888`  |
| MacOS <br>>= 10.8 (Mountain Lion)            | [Here](https://docs.docker.com/toolbox/toolbox_install_mac/)     | Docker Toolbox for Max      | Docker Quickstart Terminal | `#DOCKERIP:8888`  |
| Windows <br>10 Pro, Enterprise, or Education | [Here](https://docs.docker.com/docker-for-windows)         | Docker for Windows          | `Windows PowerShell`       | `localhost:8888`  |
| Windows <br>7, 8, 8.1, or 10 Home            | [Here](https://docs.docker.com/toolbox/toolbox_install_windows/) | Docker Toolbox for Windows  | Docker Quickstart Terminal | `#DOCKERIP:8888`  |

### Udacity's Self-Driving Car Simulator

Download the precompiled 

| OS | Link |
|:---:|:---:|


https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983385_beta-simulator-mac/beta-simulator-mac.zip