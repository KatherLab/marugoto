FROM nvcr.io/nvidia/pytorch:22.11-py3 AS develop
COPY requirements.txt /marugoto/requirements.txt
RUN pip install -r /marugoto/requirements.txt
WORKDIR /workspace

FROM develop AS deploy
WORKDIR /marugoto
COPY . /marugoto
ENTRYPOINT [ "python3", "-m" ]
