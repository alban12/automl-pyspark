import sys

import findspark
findspark.init()

from pyspark.sql import SparkSession

train_data_path = sys.argv[1]
model_path = sys.argv[2]

spark = SparkSession.builder.appName("som").getOrCreate() 
spark.sparkContext.addPyFile("./automl-iasd-0.1.0.tar.gz")

#from automl_iasd.feature_engineering.transformations import apply_discretization

df_train_data = spark.read.parquet(train_data_path)

delay = df_train_data.select("V4")

delay.write.parquet(model_path,mode="overwrite")

spark.stop()
