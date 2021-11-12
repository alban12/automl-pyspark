![image info](./assets/homepage.png) 

A distributed AutoML system for the IASD master's in apprenticeship thesis module.

## Installation

You will need poetry to use the project, you can install it with pip if you don't have it : 

```bash
pip install poetry 
```

You can then install the dependency from the `pyproject.toml` file with the following command:

```bash
poetry install 
```

There are 3 scenarios while using the project : 
- You run the tool on your local computer and the ressources are stored on AWS.
- You run the tool on a cluster and the ressources are stored on AWS.
- You run the tool on your local computer and the ressources are stored on it. (In development)

If you want to run the tool from your local computer, you will have to download Apache Hadoop and Apache Spark. 

Apache Spark can be installed pretty straightforwardly with pip : `pip install pyspark`.

Installing Hadoop will depend on your operating system, we will cover some basis for Linux and MacOS installation.

if you use Linux, you will have to do : 
if you use MacOS, you will have to do : 


You will then need to set up your computer as a [single node cluster]. (https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/SingleCluster.html)

If you run the tool on a EMR cluster, you will have to : bootstrap 

Finally, if you wish to use the tool with AWS integration, you can use the scripts in `./automl-iasd/cloud-deployment`

## Usage

To use the project, you will first need to get on an appropriate virtual environment. This can be done with the following command : 

```bash
poetry shell 
```

The tool is launched from the `automl_controller_process.py` script.
It has 5 arguments : 
- dataset : Which is one of the provided datasets in the datasets folder.
- label_column_name : Which is the name of column to predict in that label
- task : Which is the type of task to perform ("Classification", "Regression")
- budget : Which is the budget allocate to the search of the best model.

Assuming that you are in the main folder and that you want to predict the Delay for the `airlines` dataset, you can then run it with : 

```bash
python AutoML-IASD/run_pipeline\ 
	--dataset airlines\ 
	--label_column_name Delay\
	--task classification\ 
	--budget 2
```

## Licence

