# Module 4 - Building your own model
## Module Objective
It's now time to put what you've learned into action and experiment with some of your new found knowledge by creating your own model and using your preferred Deep learning framework. The overarching objective for this module is to have two trained SageMaker Endpoints in order to choose which will be the best model for __RoboStig__ to use. Since you already have a trained an optimized NVIDIA, model from the first three modules, you will now create a second one. How you accomplish this task is at your own discretion.

Below are some suggested approaches on how to accomplish the module objective. Additionally, we suggest you __focus__ on leveraging SageMaker's features for training and testing your model and not the actual model itself. To this end, you may want to look at an alternative to the NVIDIA model, from [comma.ai](https://github.com/commaai/research). This model is based on their research paper, [Learning a Driving Simulator](https://arxiv.org/pdf/1608.01230.pdf).

Before choosing one of the options, __REMEMBER__ that you are allocated roughly 3 hours for this module, so ensure that the option you select fits within this timeframe.

### Option 1 (Expert): Build your own model, using your framework of choice, by bringing your own container.

__Blah blah blah__
<!--Explain the following:
1. Why are we doing this -> flexibility of SageMaker to support framework of preference (Keras and Tensorflow) <- might want to also mention that at the time of writing, Keras support MXNet as a backend.
2. Talk about some different methods for preprocessing and color formats.-->

Here are some suggested references that may help you:
- [SageMaker Examples](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/hyperparameter_tuning/keras_bring_your_own) for bringing your own model using the [Keras](https://keras.io) framework.
- [SageMaker Examples]((https://github.com/awslabs/amazon-sagemaker-examples/tree/master/advanced_functionality/scikit_bring_your_own)) for bringing your own container.
- Now that the [Keras](https://keras.io) framework also supports [MXNet](https://aws.amazon.com/blogs/machine-learning/apache-mxnet-incubating-adds-support-for-keras-2/), you make want to leverage your accomplishments from the first three modules with [Keras](https://keras.io). Additionally, this [blog](https://medium.com/datreeio/training-with-keras-mxnet-on-amazon-sagemaker-43a34bd668ca) post from an AWS Solution Architect could be helpful.

### Option 2 (Advanced): Build your model using the other SageMaker pre-built frameworks.

### Option 3 (Intermediate): Apply a new model to the existing methodology.

### Option 4 (Novice): Use a pre-built custom container.
If none of the above approaches appeals to you, we have already built a custom container for you to leverage as a second endpoint. To proceed, follow the [instructions](./SageMakerBYOC.ipynb) on how to train the model and build the second endpoint using a pre-build container.


__blah blah blah; TIME CONSTRAINTS__

<!--Talk about the benefits of local testing since there is no build in Conda environment for Keras (at time of writing) as well as not wasting resources for training on AWS <- need to rethink this due to GPU performance considerations on local notebook instance.-->
