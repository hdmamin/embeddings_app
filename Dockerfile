FROM frolvlad/alpine-python-machinelearning

LABEL maintainer=@hmamin

WORKDIR /app

COPY requirements_docker.txt /app

RUN pip3 install -r requirements_docker.txt && \
    gdown https://drive.google.com/uc?id=1c0v0IDWfBFyVneRcjjC5BkOjITGSiD4_ -O emb.pkl

COPY . /app

CMD ["sh", "-c", "python3 app.py"]
