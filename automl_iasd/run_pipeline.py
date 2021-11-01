from IASDAutoML import IASDAutoML
from pyspark.sql import SparkSession
import click 


@click.command()
@click.option('--dataset', default="albert", help='Number of greetings.')
@click.option('--label_column_name', prompt="The name of the column to predict", help='The name of the label column')
@click.option('--task', default="classification", help='Number of greetings.')
@click.option('--budget', default=10000, help='The budget in minutes allowed for the run.')
@click.option('--verify_features_type', default=False, help='Number of greetings.')
@click.option('--save_model_path', default=".", help='The path where to store the created model.')
@click.option('--training_only', default=False, help='State if the provided dataset is only for training purpose.')
def run_pipeline(dataset, label_column_name, task, verify_features_type, save_model_path, budget, training_only):
    """Run the pipeline from the dataset in the provided framework datasets and save the best model for it in the given location."""
    spark = SparkSession \
    .builder \
    .appName("IASDAutoML") \
    .config("spark.driver.memory", "14g") \
    .getOrCreate()

    dataframe = spark.read.parquet(f"../datasets/{dataset}.parquet/",inferSchema=True, header=True)
    pipeline = IASDAutoML(budget=budget, dataframe=dataframe, label_column_name=label_column_name, task=task, training_only=training_only)
    best_model, score = pipeline.run_with_hyperopt()
    #best_model.save(f"{save_model_path}")

if __name__ == '__main__':
    run_pipeline()
