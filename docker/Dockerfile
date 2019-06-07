FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
MAINTAINER Katherine Crowson <crowsonkb@gmail.com>

# Install Ubuntu packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    curl \
    git \
    libatlas-base-dev \
    libboost-all-dev \
    libboost-python-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libprotoc-dev \
    protobuf-compiler \
    vim

# Anaconda
WORKDIR /tmp
RUN curl -O https://repo.continuum.io/archive/Anaconda3-2019.03-Linux-x86_64.sh && \
    bash Anaconda3-2019.03-Linux-x86_64.sh -b && \
    rm Anaconda3-2019.03-Linux-x86_64.sh
ENV PATH=/root/anaconda3/bin:"$PATH"

# Anaconda packages
RUN conda install -y -c intel aiohttp icc_rt tbb

# Caffe
WORKDIR /root/caffe
RUN git clone --depth 1 https://github.com/BVLC/caffe .
COPY Makefile.config .
RUN make -j"$(nproc)" all pycaffe test && make distribute

# style_transfer
WORKDIR /root/style_transfer
RUN git clone https://github.com/crowsonkb/style_transfer .
RUN ./download_models.sh
RUN pip install -r requirements.txt
COPY config.py .

EXPOSE 8000
CMD /bin/bash
