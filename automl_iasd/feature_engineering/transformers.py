from pyspark.sql.functions import when
from pyspark import keyword_only
from pyspark.ml.pipeline import Transformer
from pyspark.ml.param.shared import HasInputCol, HasInputCols, HasOutputCol, HasOutputCols, Param

class ArithmeticTransformer(Transformer, HasInputCols, HasOutputCols):
    """A transformer that apply all arithmetic operations."""

    @keyword_only
    def __init__(self, inputCols=None, outputCols=None, n=None):
        """Initialize."""
        super(ArithmeticTransformer, self).__init__()
        self.n = Param(self, "n", "")
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

class GroupByThenTransformer(Transformer, HasInputCols, HasOutputCols):
    """A transformer that apply a group by on a categorial column and a aggregation on the numeric column."""

    @keyword_only
    def __init__(self, inputCols=None, outputCols=None, n=None):
        """Initialize."""
        super(GroupByThenTransformer, self).__init__()
        self.n = Param(self, "n", "")
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
        in_cols = self.getInputCols()

        # Min 
        grouped_by_then_df = dataframe.groupBy(in_cols[0]).min(in_cols[1]).withColumnRenamed(f"min({in_cols[1]})",f"group_by_{in_cols[0]}_{out_cols[0]}_{in_cols[1]}")
        dataframe = dataframe.join(grouped_by_then_df, [in_cols[0]])

        # Max 
        grouped_by_then_df = dataframe.groupBy(in_cols[0]).max(in_cols[1]).withColumnRenamed(f"max({in_cols[1]})",f"group_by_{in_cols[0]}_{out_cols[1]}_{in_cols[1]}")
        dataframe = dataframe.join(grouped_by_then_df, [in_cols[0]])

        # Avg 
        grouped_by_then_df = dataframe.groupBy(in_cols[0]).avg(in_cols[1]).withColumnRenamed(f"avg({in_cols[1]})",f"group_by_{in_cols[0]}_{out_cols[2]}_{in_cols[1]}")
        dataframe = dataframe.join(grouped_by_then_df, [in_cols[0]])

        # Count 
        if f"group_by_{in_cols[0]}_count" not in dataframe.schema.names:
            grouped_by_then_df = dataframe.groupBy(in_cols[0]).count().withColumnRenamed(f"count",f"group_by_{in_cols[0]}_{out_cols[3]}")
            dataframe = dataframe.join(grouped_by_then_df, [in_cols[0]])

        return dataframe