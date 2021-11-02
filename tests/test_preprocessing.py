import pytest
import pyspark.sql.functions as F

from automl_iasd import __version__
from automl_iasd.transformations import remove_non_word_characters

from chispa import *


def test_version():
    assert __version__ == '0.1.0'


def test_remove_non_word_characters(spark):
    data = [
        ("jo&&se", "jose"),
        ("**li**", "li"),
        ("#::luisa", "luisa"),
        (None, None)
    ]
    df = spark.createDataFrame(data, ["name", "expected_name"])\
        .withColumn("clean_name", remove_non_word_characters(F.col("name")))
    assert_column_equality(df, "clean_name", "expected_name")


def test_infer_proper_class():
    pass

