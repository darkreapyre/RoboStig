# RoboStig - Leveraging Amazon SageMaker to build an Autonomous Self-driving Car

### Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
    * [Required Knowled](#required-knowledge)
    * [AWS Account](#aws-account)
    * [Docker](#docker)
    * [Udacity's Self-Driving Car Simulator](#udacitys-self-driving-car-simulator)
3. [Getting Started](#getting-started)

## Overview

If you're familiar with the British motoring television series [Top Gear](https://en.wikipedia.org/wiki/Top_Gear_(2002_TV_series)), then you are very familiar with [The Stig](https://www.topgear.com/car-news/stig).

>__"The Stig is a character on the British motoring television show Top Gear. The character is a play on the anonymity of racing drivers' full-face helmets, with the running joke that nobody knows who is inside the Stig's racing suit. The Stig's primary role is setting lap times for cars tested on the show." - Wikipedia__

In this workshop, you will create your own __Stig (RoboStig)__  by way of a Machine Learning (ML) technique called, __Behavioral Cloning__. This is where an ML model learns to mimmic procedural knowledge through observation. In essence you will teach your __RoboStig__ to test drive a virtual car around a track by watching you drive the vehicle.

In addition to using a [simulator](https://github.com/udacity/self-driving-car-sim) - developed by [Udacity](https://www.udacity.com/) - that captures the observation data, you will also leverage [Amazon SageMaker](https://aws.amazon.com/sagemaker) and construct a *Machine Learning Pipeline* to do the following:

1. Explore and augment the training data.
2. Train and optimize your __RoboStig__ model with your own MXNet algorithm based on the __End-to-End Deep Learning for Self-Driving Cars__ paper from [NVIDIA](https://devblogs.nvidia.com/deep-learning-self-driving-cars/).
3. Optimize your __RoboStig__ model using SageMaker Automatic Model Tuning.
4. Demonstrate the flexibility of SageMaker by using the Keras framework to create your own __RoboStig__ estimator and inference engines.
5. Evaluate the performance of the model variants and how to scale them.

At the conclusion of this workshop, you should be able to use the best model for your __RoboStig__ to drive the vehicle autonomously around the track in the simulator, and thus cloning you behavior.

---

## Requirements

The following are the requirements to successfully complete this workshop:

1. Laptop computer.
2. Prerequisite Knowledge
- You need to to understand the Python programming language (all Python code in this workshop is based version 3.6).
- You will need to have a good understanding of Deep Learning, specifically in the area of Convolutional Neural Networks. 
- You will need to have a basic knowledge of how to use Jupyter Notebooks to analyze, describe and visualize data.
- A good understanding of the MXNet, Gluon and Keras Deep Learning frameworks is beneficial.
3. AWS Account
    In order to complete this workshop you'll need an AWS Account with access to create AWS IAM, S3 and SageMaker resources. The code and instructions in this workshop assume only one student is using a given AWS account at a time. If you try sharing an account with another student, you'll run into naming conflicts for certain resources. You can work around these by appending a unique suffix to the resources that fail to create due to conflicts, but the instructions do not provide details on the changes required to make this work.
    >__Note:__ All of the resources you will launch as part of this workshop are eligible for the AWS free tier if your account is less than 12 months old. See the [AWS Free Tier page](https://aws.amazon.com/free/) for more details. It is also recommended that you use the `us-west-2` AWS Region for the workshop.
4. Docker
    In order to test the final __RoboStig__ model, you will need to install Docker for operating system and version. Follow the *Installation Instructions* for your OS in the table below:

    | OS                                       | Installation<br>Instruction               | Docker System               | Shell                      |
    |:-----------------------------------------|:-----------------------------------------:|:----------------------------|:--------------------------:|
    | Linux                                    | [Here](https://docs.docker.com/engine/installation/linux/)           | Docker for Linux            | `bash`                     |
    | MacOS <br>>= 10.10.3 (Yosemite)              | [Here](https://docs.docker.com/docker-for-mac/)             | Docker for Mac              | `bash`                     |
    | MacOS <br>>= 10.8 (Mountain Lion)            | [Here](https://docs.docker.com/toolbox/toolbox_install_mac/)     | Docker Toolbox for Max      | Docker Quickstart Terminal |
    | Windows <br>10 Pro, Enterprise, or Education | [Here](https://docs.docker.com/docker-for-windows)         | Docker for Windows          | `Windows PowerShell`       |
    | Windows <br>7, 8, 8.1, or 10 Home            | [Here](https://docs.docker.com/toolbox/toolbox_install_windows/) | Docker Toolbox for Windows  | Docker Quickstart Terminal |
5. Udacity's Self-Driving Car Simulator
    Download the zip file for your operating system below, extract it and run the executable.
    - [Mac](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983385_beta-simulator-mac/beta-simulator-mac.zip)
    - [Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983318_beta-simulator-windows/beta-simulator-windows.zip)
    - [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983558_beta-simulator-linux/beta-simulator-linux.zip)

---

## Getting Started

Before you can start, clone the repository by following these instructions:

1. Download the Git client for your operating system:
    >__Note:__ The list below contains links to the common *git-scm* client. Feel free to use an alternative client of your choosing.
    - [Mac](https://git-scm.com/download/mac)
    - [Windows](https://git-scm.com/download/win)
    - [Linux](https://git-scm.com/download/linux)
2. Clone the repository.
```bash
    $ git clone https://github.com/darkreapyre/RoboStig
```

This workshop is divided into four modules based on the steps outlined in the [Overview](#overview). Each module and the relevant instructions (`README.md`) for completing the module can be found in `modules` directory (as highlighted below). 

```
|--modules                  # Workshop modules
|  |--0_DataCapture         # First module with instructions and relevent files
|  |  |--README.md          # Module instructions
|  |...
|--src                      # RoboStig source code
|  |--README.md
|  |...
| README.md                 # This file
```

Let's start by capturing our training data for __RoboStig__ to clone, by reading the [instructions](./modules/0_DataCapture/README.md) for __Module 0: Data Capture__.

__Good Luck and have fun!__