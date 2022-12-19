#!/bin/bash

docker build -f Dockerfile.serving -t nhl-serving:v1 .
docker build -f Dockerfile.streamlit -t nhl-streamlit:v1 .