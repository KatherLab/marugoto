FROM pytorch/pytorch AS develop
WORKDIR /resources
# RUN apt-get update \
# 	&& apt-get install -y gcc git python3-dev libopenslide0 wget
COPY requirements.txt .
RUN pip install -r requirements.txt
WORKDIR /workspace

FROM develop AS deploy
WORKDIR /marugoto
COPY . /marugoto
ENTRYPOINT [ "python3", "-m" ]
