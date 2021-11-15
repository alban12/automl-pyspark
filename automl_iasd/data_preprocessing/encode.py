from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer

def encode_categorical_features(dataframe, categorical_features):
	"""Encode categorical features and add to the column of the dataframe."""
	transformations = []

	# Ordinal encoding 
	ordinal_encoded_output_cols = ["ordinal_indexed_"+categorical_feature for categorical_feature in categorical_features]
	indexer = StringIndexer(inputCols=categorical_features, outputCols=ordinal_encoded_output_cols, handleInvalid="keep")
	dataframe = indexer.fit(dataframe).transform(dataframe)

	transformations.append(indexer)

	# One-Hot-Encoding
	one_hot_encoded_output_cols = ["one_hot_encoded_"+categorical_feature for categorical_feature in categorical_features]
	encoder = OneHotEncoder(inputCols=ordinal_encoded_output_cols,
                        outputCols=one_hot_encoded_output_cols)
	model = encoder.fit(dataframe)
	dataframe = model.transform(dataframe)

	transformations.append(encoder)

	return dataframe, one_hot_encoded_output_cols, transformations
