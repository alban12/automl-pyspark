from pyspark.sql.functions import when
from pyspark import keyword_only
from pyspark.ml.pipeline import Transformer
from pyspark.ml.param.shared import HasInputCol, HasInputCols, HasOutputCol, HasOutputCols, Param
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

class ArithmeticTransformer(Transformer, HasInputCols, HasOutputCols, DefaultParamsReadable, DefaultParamsWritable):
    """Limit Cardinality of a column."""

    @keyword_only
    def __init__(self, inputCols=None, outputCols=None, n=None):
        """Initialize."""
        super(ArithmeticTransformer, self).__init__()
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
        dataframe = dataframe.withColumn(out_cols[0], in_cols[0] + in_cols[1])

        #Substraction
        dataframe = dataframe.withColumn(out_cols[1], in_cols[0] - in_cols[1])
        dataframe = dataframe.withColumn(out_cols[2], in_cols[1] - in_cols[0])

        #Mulitplication
        dataframe = dataframe.withColumn(out_cols[3], in_cols[0] * in_cols[1])
        
        #Division
        dataframe = dataframe.withColumn(out_cols[4], when(in_cols[1] != 0, in_cols[0] / in_cols[1]).otherwise(0))
        dataframe = dataframe.withColumn(out_cols[5], when(in_cols[0] != 0, in_cols[1] / in_cols[0]).otherwise(0))

        return dataframe