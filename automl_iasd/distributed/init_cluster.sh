#!/bin/bash

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
