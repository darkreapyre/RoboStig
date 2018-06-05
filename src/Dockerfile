FROM python:3.5.2

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python3-dev \
        rsync \
        software-properties-common \
        unzip \
        libgtk2.0-0 \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh tmp/Miniconda3-latest-Linux-x86_64.sh
RUN bash tmp/Miniconda3-latest-Linux-x86_64.sh -b
ENV PATH $PATH:/root/miniconda3/bin/

COPY environment.yml .
RUN conda env create -f environment.yml

# Cleanup tarballs and downloaded package files
RUN conda clean -tp -y

# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/

# Workdir
RUN mkdir /src
WORKDIR "/src"

# Jupyter
EXPOSE 8888

# Flask Server
EXPOSE 4567

# Entrypoint
COPY run.sh /
RUN chmod +x /run.sh
ENTRYPOINT ["/run.sh"]