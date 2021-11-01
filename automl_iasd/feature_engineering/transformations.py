from pyspark.ml.feature import QuantileDiscretizer, PolynomialExpansion
from pyspark.sql.functions import col
from pyspark.sql.functions import when
from pyspark.ml.feature import VectorAssembler
from automl_iasd.feature_engineering.transformers import ArithmeticTransformer, GroupByThenTransformer

def apply_discretization(dataframe, columns):
	"""Apply discretization."""
	output_columns = ["discretized_"+col for col in columns]
	discretizer = QuantileDiscretizer(numBuckets=10, inputCols=columns, outputCols=output_columns)
	dataframe = discretizer.fit(dataframe).transform(dataframe)
	return dataframe, discretizer#, output_columns

def apply_polynomial_expansion(dataframe, columns):
	"""Apply polynomial expansion."""
	#output_columns = ["polynomial_expanded_"+col for col in columns]
	output_columns = "_".join(columns)
	polynomial_assembler = VectorAssembler(
	    inputCols=columns,
	    outputCol="vectorized_"+output_columns)

	dataframe = polynomial_assembler.transform(dataframe)

	polyExpansion = PolynomialExpansion(degree=3, inputCol="vectorized_"+output_columns, outputCol="feature_polynomially_expanded_"+output_columns)
	dataframe = polyExpansion.transform(dataframe)
	return dataframe, polynomial_assembler, polyExpansion#, output_columns


def apply_binary_transformation(dataframe, columns):
	"""Apply all binary transformations on columns passed."""

	column_name = "_".join(columns)
	column_name_reverse = "_".join(columns[::-1])

	outputCols = "_".join(columns)

	arithmetic_transformer = ArithmeticTransformer(inputCols=columns, outputCols=[f"sum_{column_name}", f"difference_{column_name}", f"difference_{column_name_reverse}", f"multiplied_{column_name}", f"division_{column_name}", f"division_{column_name_reverse}"])

	dataframe = arithmetic_transformer.transform(dataframe)

	return dataframe, arithmetic_transformer


def apply_group_by_then_transformation(dataframe, categorical_column, numeric_column):
	"""Apply all unary transformations on columns passed."""

	group_by_then_transformer = GroupByThenTransformer(inputCols=[categorical_column, numeric_column], outputCols=["min", "max", "avg", "count"])
	dataframe = group_by_then_transformer.transform(dataframe)

	return dataframe, group_by_then_transformer


def create_features_column(dataframe, columns):
	"""Create a feature column from the dataframe."""
	assembler = VectorAssembler(
	    inputCols=columns,
	    outputCol="features")

	dataframe = assembler.transform(dataframe)
	return dataframe, assembler



