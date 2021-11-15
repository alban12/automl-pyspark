#!/bin/bash

# Boostrap file to set up all the nodes in a cluster with the appropriate dependencies for the AutoML jobs

sudo yum install -y git

sudo python3 -m pip install \
    matplotlib \
    hyperopt \
    boto3 \
    sagemaker \
    findspark \
    seaborn \
    sklearn \
    mlflow

sudo python3 -m pip install -Iv numpy==1.21.2

curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
source $HOME/.poetry/env 