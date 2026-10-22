import pytest
from pyspark.ml.linalg import Vectors
from automl_iasd.data_preprocessing.scaling import normalize, standardize
from chispa import assert_approx_df_equality

def test_normalize(spark):
    data = [(Vectors.dense([1.0, 2.0]),), (Vectors.dense([3.0, 4.0]),)]
    df = spark.createDataFrame(data, ["features"])

    # p=1.0 normalization: sum of absolute values should be 1.0
    # [1, 2] -> [1/3, 2/3]
    # [3, 4] -> [3/7, 4/7]
    expected_data = [
        (Vectors.dense([1.0, 2.0]), Vectors.dense([1.0/3.0, 2.0/3.0])),
        (Vectors.dense([3.0, 4.0]), Vectors.dense([3.0/7.0, 4.0/7.0]))
    ]
    expected_df = spark.createDataFrame(expected_data, ["features", "normFeatures"])

    result_df, _ = normalize(df)
    assert_approx_df_equality(result_df, expected_df, precision=1e-5)

def test_standardize(spark):
    data = [(Vectors.dense([1.0, 2.0]),), (Vectors.dense([3.0, 4.0]),)]
    df = spark.createDataFrame(data, ["features"])

    # standardize: withStd=True, withMean=False
    # Mean of squares: (1^2 + 3^2)/2 = 5, (2^2 + 4^2)/2 = 10
    # Standard deviation (unbiased):
    # col1: mean=2, var=((1-2)^2 + (3-2)^2)/(2-1) = 2, std=sqrt(2)
    # col2: mean=3, var=((2-3)^2 + (4-3)^2)/(2-1) = 2, std=sqrt(2)
    # Values: [1/sqrt(2), 2/sqrt(2)], [3/sqrt(2), 4/sqrt(2)]

    import math
    s = math.sqrt(2.0)
    expected_data = [
        (Vectors.dense([1.0, 2.0]), Vectors.dense([1.0/s, 2.0/s])),
        (Vectors.dense([3.0, 4.0]), Vectors.dense([3.0/s, 4.0/s]))
    ]
    expected_df = spark.createDataFrame(expected_data, ["features", "scaledFeatures"])

    result_df, _ = standardize(df)
    assert_approx_df_equality(result_df, expected_df, precision=1e-5)
