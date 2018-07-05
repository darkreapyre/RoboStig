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

FROM nvidia/cuda:9.0-runtime

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    git \
    bzip2 \
    libopencv-dev \
    libopenblas-dev \
    libjemalloc-dev \
    libgfortran3 \
    python-dev \
    python3-dev \
    python3-pip \
    wget \
    curl \
    nginx \
    ca-certificates

COPY model.py /opt/program/train
RUN chmod +x /opt/program/train

RUN mkdir /root/.keras
COPY keras.json /root/.keras/

RUN pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.8.0-cp36-cp36m-linux_x86_64.whl

RUN pip3 install --upgrade keras h5py pandas flask gevent gunicorn

RUN rm -rf /var/lib/apt/lists/*

RUN rm -rf /root/.cache

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY src /opt/program
WORKDIR /opt/program