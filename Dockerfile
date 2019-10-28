FROM frolvlad/alpine-python-machinelearning

LABEL maintainer=@hmamin

WORKDIR /app

COPY requirements.txt /app

RUN pip3 install -r requirements.txt

COPY . /app

RUN chmod u+x download_data.sh && \
    ./download_data.sh

CMD ["sh", "-c", "python3 app.py"]
