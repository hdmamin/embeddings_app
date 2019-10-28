# Word Embeddings App

This app lets you experiment with the 50d GloVe embeddings (https://nlp.stanford.edu/projects/glove/). The embeddings can be used to find synonyms, fill in analogies, or take on other NLP-related tasks with varying degrees of success.  
  
To run the app, clone the repo and run the shell script: 
```
git clone git@github.com:hdmamin/embeddings_app.git  
cd embeddings_app
./app.sh build  
./app.sh run
```
Then open a web browser and go to localhost:5000. (Note: at the moment, the app can be a bit slow initially, so the first one or two user submissions on each tab may take a few seconds. Currently investigating this issue, which seems to be worse inside the container.)

To stop the app, run:
```
./app.sh stop
```
