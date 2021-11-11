from pyspark.sql.functions import when
from pyspark import keyword_only
from pyspark.ml.pipeline import Transformer
from pyspark.ml.param.shared import HasInputCol, HasInputCols, HasOutputCol, HasOutputCols, Param
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

class GroupByThenTransformer(Transformer, HasInputCols, HasOutputCols, DefaultParamsReadable, DefaultParamsWritable):
    """Limit Cardinality of a column."""

    @keyword_only
    def __init__(self, inputCols=None, outputCols=None, n=None):
        """Initialize."""
        super(GroupByThenTransformer, self).__init__()
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


        print(self.getInputCols())

        print(in_cols)

        in_cols = self.getInputCols()

        # Min 
        grouped_by_then_df = dataframe.groupBy(in_cols[0]).min(in_cols[1]).withColumnRenamed(f"min({in_cols[1]})",f"group_by_{in_cols[0]}_{out_cols[0]}_{in_cols[1]}")
        dataframe = dataframe.join(grouped_by_then_df, [in_cols[0]])

        # Max 
        grouped_by_then_df = dataframe.groupBy(in_cols[0]).max(numeric_column).withColumnRenamed(f"max({in_cols[1]})",f"group_by_{in_cols[0]}_{out_cols[1]}_{in_cols[1]}")
        dataframe = dataframe.join(grouped_by_then_df, [in_cols[0]])

        # Avg 
        grouped_by_then_df = dataframe.groupBy(in_cols[0]).avg(numeric_column).withColumnRenamed(f"avg({in_cols[1]})",f"group_by_{in_cols[0]}_{out_cols[2]}_{in_cols[1]}")
        dataframe = dataframe.join(grouped_by_then_df, [in_cols[0]])

        # Count 
        if f"group_by_{categorical_column}_count" not in dataframe.schema.names:
            grouped_by_then_df = dataframe.groupBy(in_cols[0]).count().withColumnRenamed(f"count",f"group_by_{in_cols[0]}_{out_cols[3]}")
            dataframe = dataframe.join(grouped_by_then_df, [in_cols[0]])

        return dataframe, assembler