from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import Normalizer

def normalize(dataframe):
	"""Normalize the columns from the dataframe."""
	normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=1.0)
	dataframe = normalizer.transform(dataframe)
	return dataframe, normalizer

def standardize(dataframe):
	"""Standardize the columns from the dataframe."""
	scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=False)
	scalerModel = scaler.fit(dataframe)
	dataframe = scalerModel.transform(dataframe)
	return dataframe, scaler