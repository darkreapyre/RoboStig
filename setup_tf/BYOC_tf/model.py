#!/usr/bin/env python3

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

from __future__ import print_function
import os, sys, json, traceback, gzip
import numpy as np

import keras
from keras.datasets import mnist
from keras.models import Sequential, save_mxnet_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from keras.utils import multi_gpu_model

# SageMaker paths
prefix      = '/opt/ml/'
input_path  = prefix + 'input/data/'
output_path = os.path.join(prefix, 'output')
model_path  = os.path.join(prefix, 'model')
param_path  = os.path.join(prefix, 'input/config/hyperparameters.json')
data_path   = os.path.join(prefix, 'input/config/inputdataconfig.json')