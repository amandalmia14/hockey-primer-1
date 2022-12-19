#!/bin/bash

docker network create nhl-network
docker run -d -p 8501:8501 -e COMET_API_KEY=${COMET_API_KEY} --net nhl-network --name streamlit-app nhl-streamlit:v1
docker run -it -p 8080:8080 -e COMET_API_KEY=${COMET_API_KEY} --net nhl-network --name serving nhl-serving:v1