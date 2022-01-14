import boto3
import click
from pyspark.sql import SparkSession
import os

@click.command()
@click.option('--dataset_name', prompt="The name of the dataset to predict",\
				help='The name of the dataset')
@click.option('--trainset_path', prompt="The path to the trainset",\
				help='The path in s3 to the trainset.')
@click.option('--bucket_name',  prompt="The name of the bucket that holds the datasets.",\
				help='Name of the bucket that contains the dataset and that will hold the results.')
@click.option('--testset_path', default="", required="False",\
				help='The path in s3 to the testset (not mandatory if only purpose is training)')
def build_dataset_structure(dataset_name, trainset_path, bucket_name, testset_path):
	"""Send data and build paths to S3."""
	AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
	AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')

	spark = SparkSession.builder.config('spark.hadoop.fs.s3a.access.key', AWS_ACCESS_KEY).config('spark.hadoop.fs.s3a.secret.key', AWS_SECRET_KEY).getOrCreate()
	s3 = boto3.resource('s3')
	bucket = s3.Bucket(f'{bucket_name}')
	# Check if file extension is csv 
	if trainset_path.endswith("csv"):
		# Convert to parquet if needed 
		train_dataframe = spark.read.csv(f"{trainset_path}",inferSchema=True, header=True)
	else:
		train_dataframe = spark.read.parquet(f"{trainset_path}",inferSchema=True, header=True)

	if len(testset_path) > 2:
		if testset_path.endswith("csv"):
			test_dataframe = spark.read.csv(f"{testset_path}",inferSchema=True, header=True)
		else: 
			test_dataframe = spark.read.parquet(f"{testset_path}",inferSchema=True, header=True)
		test_dataframe.write.parquet(f"s3a://{bucket_name}/{dataset_name}/dataset/test_{dataset_name}.parquet")

	train_dataframe.write.parquet(f"s3a://{bucket_name}/{dataset_name}/dataset/train_{dataset_name}.parquet")
	bucket.put_object(Key=f'{dataset_name}/metafeatures/')
	bucket.put_object(Key=f'{dataset_name}/models/')
	bucket.put_object(Key=f'{dataset_name}/visualizations/')

if __name__ == '__main__':
    build_dataset_structure()