from pyspark.sql.functions import when
from pyspark import keyword_only
from pyspark.ml.pipeline import Transformer
from pyspark.ml.param.shared import HasInputCol, HasInputCols, HasOutputCol, HasOutputCols, Param
from automl_iasd.data_preprocessing.util import get_most_frequent_feature_value
import pyspark.sql.functions as F
from pyspark.sql import functions as F, types as T
from sklearn.ensemble import IsolationForest
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

class CategoricalImputer(Transformer, HasInputCols, HasOutputCols, DefaultParamsReadable, DefaultParamsWritable):
    """Limit Cardinality of a column."""

    @keyword_only
    def __init__(self, inputCols=None, outputCols=None, n=None):
        """Initialize."""
        super(CategoricalImputer, self).__init__()
        self.n = Param(self, "n", "Cardinality upper limit.")
        self._setDefault(n=25)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCols=None, n=None):
        """Get params."""
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setN(self, value):
        """Set cardinality limit."""
        return self._set(n=value)

    def getN(self):
        """Get cardinality limit."""
        return self.getOrDefault(self.n)

    def _transform(self, dataframe):
        """Do transformation."""
        out_cols = self.getOutputCols()
        in_cols = dataframe[self.getInputCols()]

        #Addition
        if out_cols[0] == "most_common" or (len(out_cols) > 1 and out_cols[1] == "Unknown_filling"):
            for column in self.getInputCols():
                most_common_feature_value = get_most_frequent_feature_value(dataframe, column)
                dataframe = dataframe.withColumn(f"most_common_{column}", F.when(F.col(f"{column}").isNull() | F.isnan(F.col(f"{column}")) | F.col(f"{column}").contains('?') | F.col(f"{column}").contains('None')| F.col(f"{column}").contains('Null') | (F.col(f"{column}") == ''),most_common_feature_value).otherwise(dataframe[f"{column}"]))
         
        if out_cols[0] == "Unknown_filling" or (len(out_cols) > 1 and out_cols[1] == "Unknown_filling"):
            for column in self.getInputCols():
                dataframe = dataframe.withColumn(f"Unknown_filling_{column}", F.when(F.col(f"{column}").isNull() | F.isnan(F.col(f"{column}")) | F.col(f"{column}").contains('?') | F.col(f"{column}").contains('None')| F.col(f"{column}").contains('Null') | (F.col(f"{column}") == ''),"Unkown").otherwise(dataframe[f"{column}"]))

        return dataframe