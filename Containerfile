FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime AS base
WORKDIR /resources
COPY requirements.txt .
RUN set -eux; \
    apt-get update; \
    apt-get install -y gcc git python3-dev libopenslide0 wget; \
    pip install -r requirements.txt

from base as develop
RUN set -eux; \
    apt-get update; \
    apt-get install -y vim; \
    pip install black mypy

FROM develop AS deploy
WORKDIR /marugoto
COPY . /marugoto
ENTRYPOINT [ "python3", "-m" ]
