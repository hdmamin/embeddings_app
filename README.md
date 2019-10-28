# Word Embeddings App

This app lets you experiment with the 50d GloVe embeddings (https://nlp.stanford.edu/projects/glove/). The embeddings can be used to find synonyms, fill in analogies, or take on other NLP-related tasks with varying degrees of success.  

First clone the app and enter its working directory.

```
git clone git@github.com:hdmamin/embeddings_app.git  
cd embeddings_app
```

For reasons I'm still investigating, the app seems to be a bit sluggish when running inside the Docker container (this mostly affects the first one or two user submissions on each tab, particularly the plotting tab). If running the app outside the container, you must first install packages from requirements.txt (either locally or inside a virtual environment), then run the following commands to download the data and then run the app: 

```
./app.sh download  
python app.py
```
 
To run the Dockerized app instead, clone the repo and run the following commands: 

```
cd embeddings_app  
./app.sh build  
./app.sh run
```

Then open a web browser and go to localhost:5000.
 
To stop the app, run:
```
./app.sh stop
```
