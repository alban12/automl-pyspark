![image info](./assets/homepage.png) 

A distributed AutoML system for the IASD master's thesis module in apprenticeship.

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
- You run the tool on your local computer and the ressources are stored on it. (In development)
- You run the tool on a cluster and the ressources are stored on AWS.

If you want to run the tool from your local computer, you will have to download Apache Hadoop and Apache Spark. 

Apache Spark can be installed pretty straightforwardly with pip : `pip install pyspark`.

Installing Hadoop will depend on your operating system.

You will then need to set up your computer as a [single node cluster](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/SingleCluster.html).

You will need to set your AWS_ACCESS_KEY and AWS_SECRET_KEY as environment variables and create a bucket to contain the dataset.

If you run the tool on a EMR cluster, you will have to make sure you include as a bootstrap action, the file : `automl-iasd\cloud-deployment\init_cluster.sh`

This will ensure that all dependencies are installed on all workers. 

Once, the cluster is running. You will need to connect to it through `ssh` and you can then clone this repository. 

## Usage

To use the project, you will first need to get on an appropriate virtual environment. This can be done with the following command : 

```bash
poetry shell 
```

The tool is launched from the `automl_controller_process.py` script.
It has 4 mandatory arguments : 
- dataset : Which is the name of the dataset set in the AWS bucket.
- label_column_name : Which is the name of column to predict in that label
- task : Which is the type of task to perform ("classification", "multiclass_classification", "regression")
- bucket_name :Which is the name of the bucket that contains the dataset and that will hold the results.

And it has 7 optional arguments : 
- budget : which is the budget allocate to the search of the best model. (default is 3)
- training_only : which state if the provided dataset is only for training purpose.
- iam_role : which is the AWS Ressource name of the role created for the feature store.
- usable_memory : which is, If run in local, the memory size that the program can use. A size too small might gives out of space error (default size is 4g)
- on_aws_cluster : Which sate if the script is ran from an AWS cluster. 

Assuming that you are in the main folder and that you want to predict the Delay for the `airlines` dataset, you can then run it with : 

```bash
python AutoML-IASD/automl_controller_process.py\ 
	--dataset=airlines\
	--label_column_name Delay\
	--task classification\ 
	--bucket_name=my_bucket_for_automl\
	--budget=2
```

Now if you wanted to predict the SalePrice for the `HousePrice` dataset, you would run : 

```bash
python AutoML-IASD/automl_controller_process.py\ 
	--dataset=house-prices\
	--label_column_name=SalePrice\
	--task=regression\ 
	--bucket_name=my_bucket_for_automl\
	--training_only=True\
```


## Licence

