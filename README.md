![image info](./assets/homepage.png) 

A distributed AutoML system for the [IASD](https://www.lamsade.dauphine.fr/wp/iasd/en/) thesis module.

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
- You run the tool on your local computer and the datasets are stored on AWS.
- You run the tool on your local computer and the datasets are stored on it. (WIP)
- You run the tool on a cluster and the datasets are stored on AWS.

If you want to run the tool from your local computer, you will need to make your computer act like a single node cluster, for which, you will have to download Apache Hadoop and Apache Spark. 

Apache Spark can be installed pretty straightforwardly with pip : `pip install pyspark`.

Installing Hadoop will depend on your operating system.

You can then set up your computer as a [single node cluster](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/SingleCluster.html).

In order to connect with the AWS resources, you will need to set your AWS_ACCESS_KEY and AWS_SECRET_KEY as environment variables and create a bucket to contain the datasets.

If choose to run the tool on a EMR cluster, you should pick as a software configuration : (Release: `emr-5.35.0`, Applications: `Spark: Spark 2.4.8 on Hadoop 2.10.1 YARN and Zeppelin 0.10.0`). Then, you will have to make sure you include, as a bootstrap action, the file : `automl-iasd\cloud-deployment\init_cluster.sh`. This will ensure that all dependencies are installed on each workers so the tool can dispatch load between each nodes. 

Once, the cluster is running. You will need to connect to the master node through `ssh`. 
Once it is done, all required softwares should be installed so you can clone this repository in it and launch a ML job with the tool. 

## Usage

Once the repository is cloned on the node, you will first need to set an appropriate virtual environment. This can be done with the following command : 

```bash
poetry shell 
```

The tool is then launched from the `automl_controller_process.py` script.
It has 4 mandatory arguments : 
- dataset : Which is the name of the dataset set in the AWS bucket.
- label_column_name : Which is the name of the column to predict 
- task : Which is the type of task to perform ("classification", "multiclass_classification", "regression")
- bucket_name : Which is the name of the bucket that contains the dataset and that will hold the results.

And it has 7 optional arguments : 
- budget : which is the budget allocate to the search of the best model. It is allocated with a weight for each step. (default is 3)
- training_only : which state if the provided dataset is only for training purpose. If "False" (default value), a portion (80%) will be used for testing the model.
- iam_role : which is the AWS Ressource name of the role created for the feature store.
- usable_memory : which is, if run in local, the memory size that the program can use. A size too small might gives out of space error (default size is 4g)
- on_aws_cluster : which sate if the script is ran from an AWS cluster. 

Assuming that you are in the main folder and that you want to predict the "Delay" columb for the `airlines` dataset, you can then run it with : 

```bash
python AutoML-IASD/automl_controller_process.py\ 
	--dataset=airlines\
	--label_column_name Delay\
	--task classification\ 
	--bucket_name=my_bucket_for_automl\
	--budget=2
```

Now, if you wanted to predict the "SalePrice" column for the `HousePrice` dataset, you would run : 

```bash
python AutoML-IASD/automl_controller_process.py\ 
	--dataset=house-prices\
	--label_column_name=SalePrice\
	--task=regression\ 
	--bucket_name=my_bucket_for_automl\
	--training_only=True\
```


## Licence

Apache License 2.0
