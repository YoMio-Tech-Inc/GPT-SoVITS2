FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

WORKDIR /workspaces

RUN sed -i 's/http:\/\/\(archive\|security\).ubuntu.com/http:\/\/mirrors.aliyun.com/g' /etc/apt/sources.list

RUN apt update

ARG DEBIAN_FRONTEND=noninteractive

RUN apt install -y ffmpeg

RUN apt install -y cmake build-essential python3-dev libsox-dev

RUN apt install -y python3-pip

RUN pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY requirements.txt /workspaces

RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . /workspaces