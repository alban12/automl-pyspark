from pyspark.sql.functions import countDistinct

def get_max_count_distinct(dataframe, columns):
	"""Compute the maximum number of different values for the given columns."""
	max_count_distinct = 0
	for column in columns:
		count_distinct = dataframe.select(countDistinct(f"{column}")).head()[0]
		if count_distinct>max_count_distinct:
			max_count_distinct = count_distinct
	return max_count_distinct+1