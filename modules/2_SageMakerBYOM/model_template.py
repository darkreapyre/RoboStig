# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

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

# BLAH BLLA BLAH

def build_model():
    """
    Create the NVidia Model.
    """
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Lambda(lambda x: x/127.5-1.0)) #Normalization
        # Complete the rest

    return net

def load_data(f_path):
    """
    Retrieves and loads the training/testing data from S3.
    
    Arguments:
    f_path -- Location for the training/testing input dataset.
    
    Returns:
    Resized training and testing data along with training and testing labels.
    """

    return None

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