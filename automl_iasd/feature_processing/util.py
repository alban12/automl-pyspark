import pyspark.sql.functions as F
from pyspark.sql.functions import monotonically_increasing_id

def get_numeric_columns_with_null_values(dataframe):
	"""Return the numeric columns wiht null values."""
	# Count the number of null in each columns 
	null_counts = dataframe.select([F.count(F.when(F.col(c[0]).isNull() | F.isnan(F.col(c[0])) | F.col(c[0]).contains('?') | F.col(c[0]).contains('None')| F.col(c[0]).contains('Null') | (F.col(c[0]) == ''), c[0])).alias(c[0]) for c in dataframe.dtypes if c[1].startswith('double') or c[1].startswith('int')]).collect()[0].asDict()
	numeric_columns_with_nulls = [k for k, v in null_counts.items() if v > 0]
	return numeric_columns_with_nulls

def get_categorical_columns_with_null_values(dataframe):
	"""Return the numeric columns wiht null values."""
	# Count the number of null in each columns 
	null_counts = dataframe.select([F.count(F.when(F.col(c[0]).isNull() | F.isnan(F.col(c[0])) | F.col(c[0]).contains('?') | F.col(c[0]).contains('None')| F.col(c[0]).contains('Null') | (F.col(c[0]) == ''), c[0])).alias(c[0]) for c in dataframe.dtypes if c[1].startswith('string') or c[1].startswith('bool')]).collect()[0].asDict()
	numeric_columns_with_nulls = [k for k, v in null_counts.items() if v > 0]
	return numeric_columns_with_nulls


def get_most_frequent_feature_value(dataframe, column_name):
	"""Return the most frequent feature value."""
	return dataframe.groupby(f"{column_name}").count().orderBy("count", ascending=False).first()[0]


def get_categorical_columns(dataframe):
	"""Return the categorical columns."""
	columnList = [item[0] for item in dataframe.dtypes if item[1].startswith('string') or item[1].startswith('bool')]
	return columnList

def get_numeric_columns(dataframe):
	"""Return the numerical columns."""
	columnList = [item[0] for item in dataframe.dtypes if item[1].startswith('int') or item[1].startswith('double')]
	return columnList

def add_id(dataframe):
	"""Add id column."""
	return dataframe.withColumn("id", monotonically_increasing_id())

def predict_using_broadcasts(feature1, feature2, feature3, feature4):
    """
    Scale the feature values and use the model to predict
    :return: 1 if normal, -1 if abnormal 0 if something went wrong
    """
    prediction = 0

    x_test = [[feature1, feature2, feature3, feature4]]
    try:
        x_test = SCL.value.transform(x_test)
        prediction = CLF.value.predict(x_test)[0]
    except ValueError:
        import traceback
        traceback.print_exc()
        print('Cannot predict:', x_test)

    return int(prediction)
