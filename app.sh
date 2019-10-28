#!/bin/bash

if [ $1 == "download" ]
then
    gdown https://drive.google.com/uc?id=1c0v0IDWfBFyVneRcjjC5BkOjITGSiD4_ -O emb.pkl
elif [ $1 == "build" ]
then
    docker build -t embedding-tool .
elif [ $1 == "run" ]
then
    docker run -d --rm --name emb-app  -p 5000:5000 embedding-tool
elif [ $1 == "stop" ]
then
    docker stop emb-app
else
    echo 'Invalid command. Options are "build", "stop", or "run".'
fi
