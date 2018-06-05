# RoboStig - Leveraging Amazon SageMaker to build an Autonomous Self-driving Car

## Overview

If you're familiar with the British motoring television series [Top Gear](https://en.wikipedia.org/wiki/Top_Gear_(2002_TV_series)), then you are very familiar with [The Stig](https://www.topgear.com/car-news/stig).

>__"The Stig is a character on the British motoring television show Top Gear. The character is a play on the anonymity of racing drivers' full-face helmets, with the running joke that nobody knows who is inside the Stig's racing suit. The Stig's primary role is setting lap times for cars tested on the show." - Wikipedia__

In this bootcamp, you will create your own __RoboStig__  by way of a Machine Learning (ML) technique called, __Behavioral Cloning__. This is where an ML model learns to mimmic procedural knowledge through observation. In essence you will teach your __RoboStig__ to test drive a virtual car around a track by watching you drive the vehicle.

In addition to using a [simulator](https://github.com/udacity/self-driving-car-sim) - developed by [Udacity](https://www.udacity.com/) - that captures the observation data, you will also leverage [Amazon SageMaker](https://aws.amazon.com/sagemaker) and construct a *Machine Learning Pipeline* to:

1. Explore and augment the training data.
2. Train and optimize your __RoboStig__ model using SageMaker's built-in [image classification algorithm](https://docs.aws.amazon.com/sagemaker/latest/dg/image-classification.html).
3. Train and optimize your __RoboStig__ model using your own algorithm based on the __End-to-End Deep Learning for Self-Driving Cars__ paper from [NVIDIA](https://devblogs.nvidia.com/deep-learning-self-driving-cars/).
4. Evaluate the performance of the model variants.
5. Use your __RoboStig__ model to drive the vehicle autonomously around the track in the simulator, thus cloning you behavior.