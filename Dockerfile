FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

WORKDIR /workspaces

RUN apt update

ARG DEBIAN_FRONTEND=noninteractive

RUN apt install -y ffmpeg

RUN apt install -y cmake build-essential python3-dev libsox-dev

RUN apt install -y python3-pip

RUN pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY requirements.txt /workspaces

RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . /workspaces