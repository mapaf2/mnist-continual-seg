ARG CUDA=11.1
ARG PYTHON_VERSION=3.7.13
ARG TORCH_VERSION=1.9.0
ARG CUDNN="8"

FROM pytorch/pytorch:${TORCH_VERSION}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV FORCE_CUDA="1"
ENV MMCV_WITH_OPS="1"
ARG MMCV_VERSION="==2.0.0"
ARG MMENGINE_VERSION="==0.7.2"
ARG ONNXRUNTIME_VERSION="1.8.1"
ARG WKDIR="/root/mnist_continual_seg"
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR ${WKDIR}
COPY . ${WKDIR}

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC \
    && apt-get update \
    && apt-get install -y \
       wget \
       ffmpeg \
       libsm6 \
       libxext6 \
       git \
       ninja-build \
       libglib2.0-0 \
       libsm6 \
       libxrender-dev \
       libxext6 \
       cmake \
       pip install matplotlib tensorflow--gpu==2.4.1 pandas tabulate

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real/:$LD_LIBRARY_PATH
ENV CUDNN_DIR=$(pwd)/cuda
ENV LD_LIBRARY_PATH=$CUDNN_DIR/lib64:$LD_LIBRARY_PATH