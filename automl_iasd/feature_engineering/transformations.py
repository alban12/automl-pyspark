from pyspark.ml.feature import QuantileDiscretizer, PolynomialExpansion
from pyspark.sql.functions import col
from pyspark.sql.functions import when
from pyspark.ml.feature import VectorAssembler

def apply_discretization(dataframe, columns):
	"""Apply discretization."""
	output_columns = ["discretized_"+col for col in columns]
	discretizer = QuantileDiscretizer(numBuckets=10, inputCols=columns, outputCols=output_columns)
	dataframe = discretizer.fit(dataframe).transform(dataframe)
	return dataframe#, output_columns

def apply_polynomial_expansion(dataframe, columns):
	"""Apply polynomial expansion."""
	#output_columns = ["polynomial_expanded_"+col for col in columns]
	output_columns = "_".join(columns)
	assembler = VectorAssembler(
	    inputCols=columns,
	    outputCol="vectorized_"+output_columns)

	dataframe = assembler.transform(dataframe)

	polyExpansion = PolynomialExpansion(degree=3, inputCol="vectorized_"+output_columns, outputCol="feature_polynomially_expanded_"+output_columns)
	dataframe = polyExpansion.transform(dataframe)
	return dataframe#, output_columns


def apply_binary_transformation(dataframe, columns):
	"""Apply all binary transformations on columns passed."""

	column_name = "_".join(columns)
	column_name_reverse = "_".join(columns[::-1])

	#Addition
	dataframe = dataframe.withColumn(f"sum_{column_name}", col(f"{columns[0]}") + col(f"{columns[1]}"))

	#Substraction
	dataframe = dataframe.withColumn(f"difference_{column_name}", col(f"{columns[0]}") - col(f"{columns[1]}"))
	dataframe = dataframe.withColumn(f"difference_{column_name_reverse}", col(f"{columns[1]}") - col(f"{columns[0]}"))

	#Mulitplication
	dataframe = dataframe.withColumn(f"multiplied_{column_name}", col(f"{columns[0]}") * col(f"{columns[1]}"))
	
	#Division
	dataframe = dataframe.withColumn(f"division_{column_name}", when(col(f"{columns[1]}") != 0, col(f"{columns[0]}") / col(f"{columns[1]}")).otherwise(0))
	dataframe = dataframe.withColumn(f"division_{column_name_reverse}", when(col(f"{columns[0]}") != 0, col(f"{columns[1]}") / col(f"{columns[0]}")).otherwise(0))

	return dataframe


def apply_group_by_then_transformation(dataframe, categorical_column, numeric_column):
	"""Apply all unary transformations on columns passed."""

	# Min 
	grouped_by_then_df = dataframe.groupBy(categorical_column).min(numeric_column).withColumnRenamed(f"min({numeric_column})",f"group_by_{categorical_column}_min_{numeric_column}")
	dataframe = dataframe.join(grouped_by_then_df, [categorical_column])

	# Max 
	grouped_by_then_df = dataframe.groupBy(categorical_column).max(numeric_column).withColumnRenamed(f"max({numeric_column})",f"group_by_{categorical_column}_max_{numeric_column}")
	dataframe = dataframe.join(grouped_by_then_df, [categorical_column])

	# Avg 
	grouped_by_then_df = dataframe.groupBy(categorical_column).avg(numeric_column).withColumnRenamed(f"avg({numeric_column})",f"group_by_{categorical_column}_avg_{numeric_column}")
	dataframe = dataframe.join(grouped_by_then_df, [categorical_column])

	# Mean 
	grouped_by_then_df = dataframe.groupBy(categorical_column).mean(numeric_column).withColumnRenamed(f"mean({numeric_column})",f"group_by_{categorical_column}_mean_{numeric_column}")
	dataframe = dataframe.join(grouped_by_then_df, [categorical_column])

	return dataframe

