FROM nvidia/cuda:9.0-base-ubuntu16.04

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-9-0 \
        cuda-cublas-9-0 \
        cuda-cufft-9-0 \
        cuda-curand-9-0 \
        cuda-cusolver-9-0 \
        cuda-cusparse-9-0 \
        curl \
        libcudnn7=7.1.4.18-1+cuda9.0 \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-dev \
        python3-dev \
        rsync \
        software-properties-common \
        unzip \
        libgtk2.0-dev \
        git \
        wget \
        nginx \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
        h5py \
        pandas \
        scipy \
        keras \
        flask \
        gevent \
        gunicorn \
        sklearn \
        opencv-python==3.4.0.12

RUN pip --no-cache-dir install --upgrade\
    tensorflow-gpu
#    https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.8.0-cp36-cp36m-linux_x86_64.whl

# RUN ln -s -f /usr/bin/python3 /usr/bin/python#

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

RUN mkdir /root/.keras
COPY keras.json /root/.keras/

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
RUN chmod +x /opt/program/train
RUN chmod +x /opt/program/serve
WORKDIR /opt/program