# Bring Your Own Framework Container to SageMaker
__blah blah blah__
<details><summary><b>Note to self</b></summary><p>
Explain the following:
1. Why are we doing this -> flexibility of SageMaker to support framework of preference (Keras and Tensorflow) <- might want to also mention that at the time of writing, Keras support MXNet as a backend.
2. Show that it's based on the [SageMaker Examples](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/hyperparameter_tuning/keras_bring_your_own) as well as [this](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/advanced_functionality/scikit_bring_your_own) and [Julien Simon](https://medium.com/datreeio/training-with-keras-mxnet-on-amazon-sagemaker-43a34bd668ca).
3. Talk about the benefits of local testing since there is no build in Conda environment for Keras (at time of writing) as well as not wasting resources for training on AWS <- need to rethink this due to GPU performance considerations on local notebook instance.
</p></details>