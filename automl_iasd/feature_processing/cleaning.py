from pyspark.ml.feature import Imputer
from automl_iasd.feature_processing.util import get_most_frequent_feature_value
import pyspark.sql.functions as F
from pyspark.sql import functions as F, types as T
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

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

def remove_outliers(dataframe, numerical_columns):
	"""Remove outliers from the dataframe."""

	# TODO : Compute visualisation 
	

	# TODO :  Send the visualisation to S3 
	
	# Find the outliers 
	scaler = StandardScaler()
	classifier = IsolationForest(contamination="auto", random_state=42, n_jobs=-1)
	x_train = [i[0:len(numerical_columns)] for i in df.select(numerical_columns).collect()]

	SCL = spark_session.sparkContext.broadcast(scaler)
	CLF = spark_session.sparkContext.broadcast(clf)
	
	udf_predict_using_broadcasts = F.udf(predict_using_broadcasts, T.IntegerType())
	
	dataframe = dataframe.withColumn('prediction',
    udf_predict_using_broadcasts('feature1', 'feature2', 'feature3', 'feature4'))

	# Remove them 


	return dataframe


