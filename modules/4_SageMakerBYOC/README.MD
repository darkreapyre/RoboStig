# Module 4 - Building your own model
## Module Objective
It's now time to put what you've learned into action and experiment with some of your new found knowledge by creating your own model and using your preferred Deep learning framework. The overarching objective for this module is to have two trained SageMaker Endpoints in order to choose which will be the best model for __RoboStig__ to use. Since you already have a trained an optimized NVIDIA, model from the first three modules, you will now create a second one. How you accomplish this task is at your own discretion.

However, here are some suggestions approaches on how to accomplish the module objective:

### Option 1 (Advanced): Build your own model using your framework of choice by bringing your own container.

### Option 2 (Intermediate): Build your model using the other SageMaker pre-built frameworks. ()

### Option 3 (Novice): Apply a new model to the existing methodology.

If none of the above approaches appeals to you, we have already built a custom container for you to leverage as a second endpoint. To proceed, follow the [instructions](./SageMakerBYOC.ipynb) on how to train the model and build the second endpoint using a pre-build container.


__blah blah blah__
<details><summary><b>Note to self</b></summary><p>
Explain the following:
1. Why are we doing this -> flexibility of SageMaker to support framework of preference (Keras and Tensorflow) <- might want to also mention that at the time of writing, Keras support MXNet as a backend.
2. Show that it's based on the [SageMaker Examples](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/hyperparameter_tuning/keras_bring_your_own) as well as [this](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/advanced_functionality/scikit_bring_your_own) and [Julien Simon](https://medium.com/datreeio/training-with-keras-mxnet-on-amazon-sagemaker-43a34bd668ca).
3. Talk about the benefits of local testing since there is no build in Conda environment for Keras (at time of writing) as well as not wasting resources for training on AWS <- need to rethink this due to GPU performance considerations on local notebook instance.
</p></details>