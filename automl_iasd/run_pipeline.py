from IASDAutoML import IASDAutoML
from pyspark.sql import SparkSession
import click 

@click.command()
@click.option('--dataset', default="albert", help='Number of greetings.')
@click.option('--label_column_name', prompt="The name of the column to predict", help='The name of the label column')
@click.option('--task', default="classification", help='Number of greetings.')
@click.option('--budget', default=10000, help='The budget in minutes allowed for the run.')
@click.option('--verify_features_type', default=False, help='Number of greetings.')
@click.option('--save_model_path', default="with_date_best_model", help='The path where to store the created model.')
def run_pipeline(dataset, label_column_name, task, verify_features_type, save_model_path, budget):
    """Run the pipeline from the dataset in the provided framework datasets and save the best model for it in the given location."""
    spark = SparkSession \
    .builder \
    .appName("IASDAutoML") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

    dataframe = spark.read.parquet(f"../datasets/{dataset}.parquet/",inferSchema=True, header=True)
    pipeline = IASDAutoML(budget=budget, dataframe=dataframe, label_column_name=label_column_name, task=task)
    best_model = pipeline.run()
    #best_model.save(f"{save_model_path}")

if __name__ == '__main__':
    run_pipeline()