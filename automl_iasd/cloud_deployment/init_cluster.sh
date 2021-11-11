#!/bin/bash

# Boostrap file to set up all the nodes in a cluster with the appropriate dependencies for the AutoML jobs

sudo yum install -y git

sudo python3 -m pip install \
    matplotlib \
    hyperopt \
    boto3 \
    numpy \
    findspark \
    seaborn \
    sklearn \
    mlflow

curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
source $HOME/.poetry/env 