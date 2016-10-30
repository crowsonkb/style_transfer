FROM ubuntu:14.04
MAINTAINER Katherine Crowson <crowsonkb@gmail.com>

RUN apt-get update && apt-get install -y --no-install-recommends \
    automake \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libboost-all-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libhdf5-serial-dev \
    libtool \
    unzip

# Intel MKL BLAS library
WORKDIR /tmp
COPY silent.cfg .
RUN curl -O http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/9662/l_mkl_2017.0.098.tgz && \
    tar xzvf l_mkl_2017.0.098.tgz && \
    rm l_mkl_2017.0.098.tgz && \
    l_mkl_2017.0.098/install.sh --silent silent.cfg && \
    rm -rf l_mkl_2017.0.098
RUN echo '. /opt/intel/mkl/bin/mklvars.sh intel64' >> /root/.bashrc

# Python environment
WORKDIR /tmp
RUN curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/root/miniconda3/bin:"$PATH"
RUN conda install -y matplotlib mkl-service numpy pillow scikit-image scipy six
RUN pip install posix-ipc protobuf

# protobuf
WORKDIR /tmp/protobuf
RUN git clone https://github.com/google/protobuf .
RUN git checkout v3.1.0
RUN ./autogen.sh
RUN ./configure
RUN make -j"$(nproc)"
RUN make install
RUN ldconfig
WORKDIR /
RUN rm -rf /tmp/protobuf

# Caffe
WORKDIR /opt/caffe
RUN git clone --depth 1 https://github.com/BVLC/caffe .
COPY Makefile.config .
RUN make -j"$(nproc)" all pycaffe

# style_transfer
WORKDIR /opt/style_transfer
RUN git clone --depth 1 https://github.com/crowsonkb/style_transfer .
RUN ./download_models.sh

ENV PYTHONPATH /opt/caffe/python
EXPOSE 8000
CMD /bin/bash
