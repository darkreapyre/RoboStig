# Bring Your Own Model to SageMaker - NVIDIA Model
<details><summary><b>Note to self</b></summary><p>
Explain why the built-in image classification algorithm can't be used here.
</p></details>

[End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)
![Architecture](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png)

## Training Functions
__blah blah blah__  

<details><summary><b>Note to self</b></summary><p>
Make sure to highlight the fact that the necessary libraries have already been configured in the model template file.
</p></details>


### Building the model: `build_model()` Function
__Blah blah blah__

```python
def build_model():
    """
    Create the NVidia Model.
    """
    net = None

    return net
```

<details><summary><b>Solution (Click to expand)</b></summary><p>
```
def build_model():
    """
    Create the NVidia Model.
    """
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Lambda(lambda x: x/127.5-1.0)) #Normalization
        net.add(gluon.nn.Conv2D(channels=24, kernel_size=(5, 5), strides=(2, 2), padding=1, activation='relu'))
        net.add(gluon.nn.Conv2D(channels=36, kernel_size=(5, 5), strides=(2, 2), padding=1, activation='relu'))
        net.add(gluon.nn.Conv2D(channels=48, kernel_size=(5, 5), strides=(2, 2), padding=1, activation='relu'))
        net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, activation='relu'))
        net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, activation='relu'))
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(1164))
        net.add(gluon.nn.Activation('relu'))
        net.add(gluon.nn.Dense(100))
        net.add(gluon.nn.Activation('relu'))
        net.add(gluon.nn.Dense(50))
        net.add(gluon.nn.Activation('relu'))
        net.add(gluon.nn.Dense(10))
        net.add(gluon.nn.Activation('relu'))
        net.add(gluon.nn.Dense(1))
    net.hybridize()
    return net
```
</p></details>

### Resizing the Input Shapes: `transform()` Function
__BLAH BLAH BLAH__  

```python
def transform(x, y):
    """
    Reshape the numpy arrays as 4D Tensors for MXNet.
    
    Arguments:
    x -- Numpy Array of input images
    y -- Numpy Array of labels
    
    Returns:
    x -- Numpy Array as (NCHW).
    y -- Label as Column vector.
    """
    data  = None
    label = None
    return data, label
```
<details><summary><b>Solution (Click to expand)</b></summary><p>

```
def transform(x, y):
    """
    Reshape the numpy arrays as 4D Tensors for MXNet.
    
    Arguments:
    x -- Numpy Array of input images
    y -- Numpy Array of labels
    
    Returns:
    x -- Numpy Array as (NCHW).
    y -- Label as Column vector.
    """
    data  = x.reshape(-1, 3, 66, 200)
    label = y.reshape(-1, 1)
    return data, label
```
</p></details>

### Loading and Transforming the Datasets: `load_data()` Function
__blah blah blah__

```python
def load_data(f_path):
    """
    Retrieves and loads the training/testing data from S3.
    
    Arguments:
    f_path -- Location for the training/testing input dataset.
    
    Returns:
    Resized training and testing data along with training and testing labels.
    """

    return None
```
<details><summary><b>Solution (Click to expand)</b></summary><p>

```
def load_data(f_path):
    """
    Retrieves and loads the training/testing data from S3.
    
    Arguments:
    f_path -- Location for the training/testing input dataset.
    
    Returns:
    Pre-processed training and testing data along with training and testing labels.
    """
    train_x = np.load(f_path+'/train_X.npy')
    train_y = np.load(f_path+'/train_Y.npy')
    train_X, train_Y = transform(train_x, train_y)
    test_x = np.load(f_path+'/valid_X.npy')
    test_y = np.load(f_path+'/valid_Y.npy')
    test_X, test_Y = transform(test_x, test_y)
    return train_X, train_Y, test_X, test_Y
```
</p></details>


### Measuring the Training vs. Validation Accuracy: `accuracy()` Function
__blah blah blah; already provided__

```python
def accuracy(data_iterator, net, ctx):
    """
    Evaluates the Accuracy of the model against the Training or Testing iterator.
    
    Arguments:
    data_iterator -- Iterator.
    net -- Gluon Model.
    
    Returns:
    Accuracy of the model against the data iterator.
    """
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]
```

### Fitting the Model: `train()` Function
__blah blah blah__

```python
def train(channel_input_dirs, hyperparameters, hosts, num_gpus, output_data_dir, **kwargs):
    """
    BLAH BLAH BLAH

    """
    # Set the Context
    
    # Set Local vs. Distributed training
    if len(hosts) == 1:
        kvstore = 'device' if num_gpus > 0 else 'local'
    else:
        kvstore = 'dist_device_sync' if num_gpus > 0 else 'dist_sync'
    
    # Load hyperparameters
    
    # Load Training/Testing Data
    
    # Create Training and Test Data Iterators
    train_data = None
    test_data = None
    
    # Initialize the neural network structure
    net = build_model()
    
    # Parameter Initialization
    
    # Optimizer
    
    # Squared Error Loss Function
    square_loss = None
    
    # Train the model
    for epoch in range(epochs):
        curr_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = None
            label = None
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            curr_loss += nd.mean(loss).asscalar()
        val_accuracy = accuracy(test_data, net, ctx)
        train_accuracy = accuracy(train_data, net, ctx)
        print(
            "Epoch {}: Loss: {}; Training Accuracy = {}; Validation Accuracy = {}"\
            .format(epoch, curr_loss/len(train_data), train_accuracy, val_accuracy)
        )

    # Return the model for saving
    return None
```
<details><summary><b>Solution (Click to expand)</b></summary><p>

```
def train(channel_input_dirs, hyperparameters, hosts, num_gpus, output_data_dir, **kwargs):
    """
    blah blah blah
    """
    # Set the Context
    ctx = mx.gpu() if num_gpus > 0 else mx.cpu()
    
    # Set Local vs. Distributed training
    if len(hosts) == 1:
        kvstore = 'device' if num_gpus > 0 else 'local'
    else:
        kvstore = 'dist_device_sync' if num_gpus > 0 else 'dist_sync'
    
    # Load hyperparameters
    epochs = hyperparameters.get('epochs', 11)
    optmizer = hyperparameters.get('optmizer', 'adam')
    lr = hyperparameters.get('learning_rate', 1.0e-4)
    batch_size = hyperparameters.get('batch_size', 64)
    
    # Load Training/Testing Data
    f_path = channel_input_dirs['training']
    train_X, train_Y, test_X, test_Y = load_data(f_path)
    
    # Create Training and Test Data Iterators
    train_data = mx.gluon.data.DataLoader(
        mx.gluon.data.ArrayDataset(
            train_X,
            train_Y
        ),
        shuffle=True,
        batch_size=batch_size
    )
    test_data = mx.gluon.data.DataLoader(
        mx.gluon.data.ArrayDataset(
            test_X,
            test_Y
        ),
        shuffle=False,
        batch_size=batch_size
    )
    
    # Initialize the neural network structure
    net = build_model()
    
    # Parameter Initialization
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    
    # Optimizer
    trainer = gluon.Trainer(net.collect_params(), optmizer, {'learning_rate': lr})
    
    # Squared Error Loss Function
    square_loss = gluon.loss.L2Loss()
    
    # Train the model
    for epoch in range(epochs):
        curr_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            curr_loss += nd.mean(loss).asscalar()
        val_accuracy = accuracy(test_data, net, ctx)
        train_accuracy = accuracy(train_data, net, ctx)
        print("Epoch {}: Loss: {}; Training Accuracy = {}; Validation Accuracy = {}"\
              .format(epoch, curr_loss/len(train_data), train_accuracy, val_accuracy)
             )

    # Return the model for saving
    return net
```
</p></details>

### Saving the Model for Inference: `save()` Function
__blah blah blah; already provided__

```python
def save(net, model_dir):
    """
    Saves the trained model to S3.
    
    Arguments:
    model -- The model returned from the `train()` function.
    model_dir -- The model directory location to save the model.
    """
    print("Saving the trained model ...")
    y = net(mx.sym.var('data'))
    y.save('%s/model.json' % model_dir)
    net.collect_params().save('%s/model.params' % model_dir)
```

---
## Inference Functions
__blah blah blah__

### Model Hosting: `model_fn` Function
__blah blah blah; even though it won't be used just yet__  

<details><summary><b>Note to self</b></summary><p>
Test this function code below before completion!
</p></details>

```python
def model_fn(model_dir):
    """
    Load the Gluon model for hosting.

    Arguments:
    model_dir -- SageMaker model directory.

    Retuns:
    Gluon model
    """
    # Load the saved Gluon model
    symbol = mx.sym.load('%s/model.json' % model_dir)
    outputs = mx.sym.sigmoid(data=symbol, name='sigmoid_label')
    inputs = mx.sym.var('data')
    param_dict = gluon.ParameterDict('model_')
    net = gluon.SymbolBlock(outputs, inputs, param_dict)
    net.load_params('%s/model.params' % model_dir, ctx=mx.cpu())
    return net
```

### Prediction Result: `transform_fn()` Function
__blah blah blah__

```python
def transform_fn(net, data, input_content_type, output_content_type):
    """
    Transform input data into prediction result.

    Argument:
    net -- Gluon model loaded from `model_fn()` function.
    data -- Input data from the `InvokeEndpoint` request.
    input_content_type -- Content type of the request (JSON).
    output_content_type -- Desired content type (JSON) of the repsonse.
    
    Returns:
    JSON payload of the prediction result and content type.
    """
    # Parse the data
    parsed = loads(data)
    # Convert input to MXNet NDArray
    nda = mx.nd.array(parsed)
    output = net(nda)
    prediction = nd.argmax(output, axis=1)
    response_body = dumps(prediction.asnumpy().tolist()[0])
    return response_body, output_content_type
```

---
## Building the Estimator Script
__blah blah blah; copy your functions to `model.py`; should look as follows:__  

<details><summary><b>`model.py` (Click to expand)</b></summary><p>

```
# Import necessary libraries
import boto3
import os
import io
import logging
import datetime
import json
import mxnet as mx
import numpy as np
from json import dumps, loads
from mxnet import nd, autograd, gluon

# Set logging
logging.getLogger("requests.packages.urllib3.connectionpool").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------- #
#                            Training functions                                #
# ---------------------------------------------------------------------------- #
def train(channel_input_dirs, hyperparameters, hosts, num_gpus, output_data_dir, **kwargs):
    # Set the Context
    ctx = mx.gpu() if num_gpus > 0 else mx.cpu()
    
    # Set Local vs. Distributed training
    if len(hosts) == 1:
        kvstore = 'device' if num_gpus > 0 else 'local'
    else:
        kvstore = 'dist_device_sync' if num_gpus > 0 else 'dist_sync'
    
    # Load hyperparameters
    epochs = hyperparameters.get('epochs', 11)
    optmizer = hyperparameters.get('optmizer', 'adam')
    lr = hyperparameters.get('learning_rate', 1.0e-4)
    batch_size = hyperparameters.get('batch_size', 64)
    
    # Load Training/Testing Data
    f_path = channel_input_dirs['training']
    train_X, train_Y, test_X, test_Y = load_data(f_path)
    
    # Create Training and Test Data Iterators
    train_data = mx.gluon.data.DataLoader(
        mx.gluon.data.ArrayDataset(
            train_X,
            train_Y
        ),
        shuffle=True,
        batch_size=batch_size
    )
    test_data = mx.gluon.data.DataLoader(
        mx.gluon.data.ArrayDataset(
            test_X,
            test_Y
        ),
        shuffle=False,
        batch_size=batch_size
    )
    
    # Initialize the neural network structure
    net = build_model()
    
    # Parameter Initialization
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    
    # Optimizer
    trainer = gluon.Trainer(net.collect_params(), optmizer, {'learning_rate': lr})
    
    # Squared Error Loss Function
    square_loss = gluon.loss.L2Loss()
    
    # Train the model
    for epoch in range(epochs):
        curr_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            curr_loss += nd.mean(loss).asscalar()
        val_accuracy = accuracy(test_data, net, ctx)
        train_accuracy = accuracy(train_data, net, ctx)
        print("Epoch {}: Loss: {}; Training Accuracy = {}; Validation Accuracy = {}"\
              .format(epoch, curr_loss/len(train_data), train_accuracy, val_accuracy)
             )

    # Return the model for saving
    return net

def build_model():
    """
    Create the NVidia Model.
    """
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Lambda(lambda x: x/127.5-1.0)) #Normalization
        net.add(gluon.nn.Conv2D(channels=24, kernel_size=(5, 5), strides=(2, 2), padding=1, activation='relu'))
        net.add(gluon.nn.Conv2D(channels=36, kernel_size=(5, 5), strides=(2, 2), padding=1, activation='relu'))
        net.add(gluon.nn.Conv2D(channels=48, kernel_size=(5, 5), strides=(2, 2), padding=1, activation='relu'))
        net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, activation='relu'))
        net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, activation='relu'))
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(1164))
        net.add(gluon.nn.Activation('relu'))
        net.add(gluon.nn.Dense(100))
        net.add(gluon.nn.Activation('relu'))
        net.add(gluon.nn.Dense(50))
        net.add(gluon.nn.Activation('relu'))
        net.add(gluon.nn.Dense(10))
        net.add(gluon.nn.Activation('relu'))
        net.add(gluon.nn.Dense(1))
    net.hybridize()
    return net

def transform(x, y):
    """
    Reshape the numpy arrays as 4D Tensors for MXNet.
    
    Arguments:
    x -- Numpy Array of input images
    y -- Numpy Array of labels
    
    Returns:
    x -- Numpy Array as (NCHW).
    y -- Label as Column vector.
    """
    data  = x.reshape(-1, 3, 66, 200)
    label = y.reshape(-1, 1)
    return data, label

def load_data(f_path):
    """
    Retrieves and loads the training/testing data from S3.
    
    Arguments:
    f_path -- Location for the training/testing input dataset.
    
    Returns:
    Pre-processed training and testing data along with training and testing labels.
    """
    train_x = np.load(f_path+'/train_X.npy')
    train_y = np.load(f_path+'/train_Y.npy')
    train_X, train_Y = transform(train_x, train_y)
    test_x = np.load(f_path+'/valid_X.npy')
    test_y = np.load(f_path+'/valid_Y.npy')
    test_X, test_Y = transform(test_x, test_y)
    return train_X, train_Y, test_X, test_Y


def save(net, model_dir):
    """
    Saves the trained model to S3.
    
    Arguments:
    model -- The model returned from the `train()` function.
    model_dir -- The model directory location to save the model.
    """
    print("Saving the trained model ...")
    y = net(mx.sym.var('data'))
    y.save('%s/model.json' % model_dir)
    net.collect_params().save('%s/model.params' % model_dir)

def accuracy(data_iterator, net, ctx):
    """
    Evaluates the Accuracy of the model against the Training or Testing iterator.
    
    Arguments:
    data_iterator -- Iterator.
    net -- Gluon Model.
    
    Returns:
    Accuracy of the model against the data iterator.
    """
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

# ---------------------------------------------------------------------------- #
#                           Hosting functions                                  #
# ---------------------------------------------------------------------------- #

def model_fn(model_dir):
    """
    Load the Gluon model for hosting.

    Arguments:
    model_dir -- SageMaker model directory.

    Retuns:
    Gluon model
    """
    # Load the saved Gluon model
    symbol = mx.sym.load('%s/model.json' % model_dir)
    outputs = mx.sym.sigmoid(data=symbol, name='sigmoid_label')
    inputs = mx.sym.var('data')
    param_dict = gluon.ParameterDict('model_')
    net = gluon.SymbolBlock(outputs, inputs, param_dict)
    net.load_params('%s/model.params' % model_dir, ctx=mx.cpu())
    return net

def transform_fn(net, data, input_content_type, output_content_type):
    """
    Transform input data into prediction result.

    Argument:
    net -- Gluon model loaded from `model_fn()` function.
    data -- Input data from the `InvokeEndpoint` request.
    input_content_type -- Content type of the request (JSON).
    output_content_type -- Desired content type (JSON) of the repsonse.
    
    Returns:
    JSON payload of the prediction result and content type.
    """
    # Parse the data
    parsed = loads(data)
    # Convert input to MXNet NDArray
    nda = mx.nd.array(parsed)
    output = net(nda)
    prediction = nd.argmax(output, axis=1)
    response_body = dumps(prediction.asnumpy().tolist()[0])
    return response_body, output_content_type
```
</p></details>

---
# Next Step: Training in SageMaker
Now that you've built the Estimator Script, you will use SageMaker to to train your __RoboStig__ model. Open the [SageMakerBYOM](./SageMakerBYOM.ipynb) Notebook to start the training job.