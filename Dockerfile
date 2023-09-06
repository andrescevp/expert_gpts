FROM python:3.10.13-slim

RUN apt update && apt upgrade -y
RUN apt install software-properties-common build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev libpq-dev htop top -y

RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN python --version
RUN pip --version

COPY . /app

WORKDIR /app
