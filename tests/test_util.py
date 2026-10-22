import pytest
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, BooleanType, IntegerType, StructField, StructType
from automl_iasd.data_preprocessing.util import get_categorical_columns_with_missing_values

def test_get_categorical_columns_with_missing_values(spark):
    schema = StructType([
        StructField("col_none", StringType(), True),
        StructField("col_empty", StringType(), True),
        StructField("col_question", StringType(), True),
        StructField("col_none_str", StringType(), True),
        StructField("col_null_str", StringType(), True),
        StructField("col_bool_none", BooleanType(), True),
        StructField("col_clean", StringType(), True),
        StructField("col_numeric_none", IntegerType(), True),
    ])

    data = [
        (None, "", "?", "None", "Null", None, "clean", None),
        ("val", "val", "val", "val", "val", True, "val", 1),
    ]

    df = spark.createDataFrame(data, schema)

    result = get_categorical_columns_with_missing_values(df)

    expected = [
        "col_none",
        "col_empty",
        "col_question",
        "col_none_str",
        "col_null_str",
        "col_bool_none"
    ]

    assert sorted(result) == sorted(expected)

def test_get_categorical_columns_with_missing_values_no_cats(spark):
    schema = StructType([
        StructField("col_numeric", IntegerType(), True),
    ])

    data = [(None,), (1,)]
    df = spark.createDataFrame(data, schema)

    result = get_categorical_columns_with_missing_values(df)
    assert result == []

def test_get_categorical_columns_with_missing_values_all_clean(spark):
    schema = StructType([
        StructField("col_str", StringType(), True),
        StructField("col_bool", BooleanType(), True),
    ])

    data = [("val", True), ("other", False)]
    df = spark.createDataFrame(data, schema)

    result = get_categorical_columns_with_missing_values(df)
    assert result == []
