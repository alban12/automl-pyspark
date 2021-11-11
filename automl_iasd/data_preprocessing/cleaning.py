from pyspark.ml.feature import Imputer
from automl_iasd.data_preprocessing.util import get_most_frequent_feature_value
import pyspark.sql.functions as F
from pyspark.sql import functions as F, types as T
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from automl_iasd.data_preprocessing.CategoricalImputer import CategoricalImputer
from pyspark.ml.feature import VectorAssembler
import boto3
from pyspark.ml.feature import PCA
import matplotlib.pyplot as plt
import io

def fill_missing_values(numeric_columns_to_fill, categorical_columns_to_fill, dataframe, strategy="expand"):
	"""Fill the missing values by applying all possible transformation to each feature."""
	raw_numeric_features_name = [item[0] for item in dataframe.dtypes if item[1].startswith('double') or item[1].startswith('integer')]
	numerical_fill_strategy = ["mean", "median", "mode"]
	categorical_fill_strategy = ["most_common", "Unknown_filling"]
	transformations = []
	for strategy in numerical_fill_strategy:
		imputed_features_names = [f"imputed_{strategy}_"+feature for feature in numeric_columns_to_fill]
		imputer = Imputer(inputCols=[*numeric_columns_to_fill], outputCols=[*imputed_features_names])
		model = imputer.fit(dataframe)
		dataframe = model.transform(dataframe)
		transformations.append(imputer)

	for strategy in categorical_fill_strategy:
		if strategy == "most_common":
			for column in categorical_columns_to_fill:
				most_common_feature_value = get_most_frequent_feature_value(dataframe, column)
				dataframe = dataframe.withColumn(f"{strategy}_{column}", F.when(F.col(f"{column}").isNull() | F.isnan(F.col(f"{column}")) | F.col(f"{column}").contains('?') | F.col(f"{column}").contains('None')| F.col(f"{column}").contains('Null') | (F.col(f"{column}") == ''),most_common_feature_value).otherwise(dataframe[f"{column}"]))
			transformations.append(imputer)
		if strategy == "Unknown_filling":
			for column in categorical_columns_to_fill:
				dataframe = dataframe.withColumn(f"{strategy}_{column}", F.when(F.col(f"{column}").isNull() | F.isnan(F.col(f"{column}")) | F.col(f"{column}").contains('?') | F.col(f"{column}").contains('None')| F.col(f"{column}").contains('Null') | (F.col(f"{column}") == ''),"Unkown").otherwise(dataframe[f"{column}"]))
			transformations.append(imputer)
	return dataframe, transformations

def remove_outliers(dataframe, numerical_columns, bucket_name, label_column_name, dataset):
	"""Remove outliers from the dataframe."""

	# Marks outlier : ex : (1.0,2.4)(1) (observation with label 1), (2.3,234234.4)(-1) (observation which is an outlier => -1)

	# Doesn't work 

	# Create a features column for visualization 
	viz_features_assembler = VectorAssembler(inputCols=numerical_columns, outputCol="initial_numerical_features")
	dataframe = viz_features_assembler.transform(dataframe)

	# Reduce the dimension if necessary 
	if len(numerical_columns)>2:
	    pca = PCA(k=2, inputCol="initial_numerical_features", outputCol="pcaFeatures")
	    model = pca.fit(dataframe)
	    viz_dataframe = model.transform(dataframe).select("pcaFeatures", "Delay")
	    x_train_pc1 = viz_dataframe.select("pcaFeatures").rdd.map(lambda x: x[:][0][0]).collect()
	    x_train_pc2 = viz_dataframe.select("pcaFeatures").rdd.map(lambda x: x[:][0][1]).collect()
	else:
	    x_train_pc1 = dataframe.select(numerical_columns[0]).rdd.map(lambda x: x[:][0]).collect()
	    x_train_pc2 = dataframe.select(numerical_columns[1]).rdd.map(lambda x: x[:][0]).collect()

	y = viz_dataframe.select(label_column_name).rdd.map(lambda x: x[:][0]).collect()

	plt.figure(figsize=(20,12))
	plt.scatter(x_train_pc1, x_train_pc2, c = y)
	plt.xlabel('PC1')
	plt.ylabel('PC2')

	# Send the visualisation to S3 
	img_data = io.BytesIO()
	plt.savefig(img_data, format='png')
	img_data.seek(0)
	image = img_data.read()

	s3 = boto3.resource('s3')
	bucket = s3.Bucket(bucket_name)
	bucket.put_object(Body=image, ContentType='image/png', Key=f"{dataset}/visualizations/{dataset}-scatter-plot.png")
	
	# Find the outliers 
	# scaler = StandardScaler()
	# classifier = IsolationForest(contamination="auto", random_state=42, n_jobs=-1)
	# x_train = [i[0:len(numerical_columns)] for i in df.select(numerical_columns).collect()]

	# SCL = spark_session.sparkContext.broadcast(scaler)
	# CLF = spark_session.sparkContext.broadcast(clf)
	
	# udf_predict_using_broadcasts = F.udf(predict_using_broadcasts, T.IntegerType())
	
	# dataframe = dataframe.withColumn('prediction',
 	#    udf_predict_using_broadcasts('feature1', 'feature2', 'feature3', 'feature4'))

	# Remove them 
	return dataframe


def predict_using_broadcasts(*numerical_columns):
    """
    Scale the feature values and use the model to predict
    :return: 1 if normal, -1 if abnormal 0 if something went wrong
    """
    prediction = 0

    x_test = [numerical_columns]
    try:
        x_test = SCL.value.transform(x_test)
        prediction = CLF.value.predict(x_test)[0]
    except ValueError:
        import traceback
        traceback.print_exc()
        print('Cannot predict:', x_test)

    return int(prediction)

