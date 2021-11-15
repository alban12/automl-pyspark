from pyspark.ml.feature import VectorSlicer
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import Row
from pyspark.sql import SparkSession
import random
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

#Transformer
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType


import nltk

from pyspark import keyword_only  ## < 2.0 -> pyspark.ml.util.keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

class SumColumn(Transformer, HasInputCol, HasOutputCol):

    @keyword_only
    def __init__(self, inputCol=None, inputCol2=None, outputCol=None, stopwords=None):
        super(SumColumn, self).__init__()
        self.stopwords = Param(self, "stopwords", "")
        self._setDefault(stopwords=set())
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, stopwords=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setStopwords(self, value):
        self._paramMap[self.stopwords] = value
        return self

    def getStopwords(self):
        return self.getOrDefault(self.stopwords)

    def _transform(self, dataset):
        stopwords = self.getStopwords()

        def f(s):
            tokens = nltk.tokenize.wordpunct_tokenize(s)
            return [t for t in tokens if t.lower() not in stopwords]

        t = ArrayType(StringType())
        out_col = self.getOutputCol()
        in_col = dataset[self.getInputCol()]
        return dataset.withColumn(out_col, udf(f, t)(in_col))

class ArithmeticTransformer(Transformer, HasInputCol, HasOutputCol):
    """Limit Cardinality of a column."""

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, n=None):
        """Initialize."""
        super(ArithmeticTransformer, self).__init__()
        self.n = Param(self, "n", "Cardinality upper limit.")
        self._setDefault(n=25)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, n=None):
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
        out_col = self.getOutputCol()
        in_col = dataframe[self.getInputCol()]

        firstelement=udf(lambda v:float(v[0]),FloatType())
        secondelement=udf(lambda v:float(v[1]),FloatType())

        #column_name = "_".join(columns)
        #column_name_reverse = "_".join(columns[::-1])

        #column_name = "_"
        #column_name_reverse = "_"

        #Addition
        dataframe = dataframe.withColumn(f"sum_{in_col}", firstelement(in_col) + secondelement(in_col))

        #Substraction
        #dataframe = dataframe.withColumn(f"difference_{in_col}", col(f"{columns[0]}") - col(f"{columns[1]}"))
        #dataframe = dataframe.withColumn(f"difference_inverse_{in_col}", col(f"{columns[1]}") - col(f"{columns[0]}"))

        #Mulitplication
        #dataframe = dataframe.withColumn(f"multiplied_{in_col}", col(f"{columns[0]}") * col(f"{columns[1]}"))
        
        #Division
        #dataframe = dataframe.withColumn(f"division_{in_col}", when(col(f"{columns[1]}") != 0, col(f"{columns[0]}") / col(f"{columns[1]}")).otherwise(0))
        #dataframe = dataframe.withColumn(f"division_inverse_{in_col}", when(col(f"{columns[0]}") != 0, col(f"{columns[1]}") / col(f"{columns[0]}")).otherwise(0))

        #return dataframe.withColumn(f"sum_{column_name}", col(f"{columns[0]}") + col(f"{columns[1]}"))
        return dataframe



spark = SparkSession \
    .builder \
    .appName("IASDAutoML") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

df = spark.createDataFrame([
    Row(userFeatures=Vectors.sparse(658,[0,1,2,3,4,5,6,19,301,503,608,609,610,611,612,613,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,630,631,632,633,634,635,636,637,638,639,640,641,642,643,644,645,646,647,648,649,650,651,652,653,654,655,656,657],[7809.0,6.0,815.0,71.0,12.0,2.0,187.0,1.0,1.0,1.0,1.0,6.0,5.0,9.0,71.0,6.0,815.0,7809.0,71.0,5041.0,357911.0,6.0,426.0,30246.0,36.0,2556.0,216.0,815.0,57865.0,4108415.0,4890.0,347190.0,29340.0,664225.0,4.7159975E7,3985350.0,5.41343375E8,7809.0,554439.0,3.9365169E7,46854.0,3326634.0,281124.0,6364335.0,4.51867785E8,3.818601E7,5.186933025E9,6.0980481E7,4.329614151E9,3.65882886E8,4.9699092015E10,4.76196576129E11,3044.0,7814.0,6099.099633859992,50254.0,7777.0,7809.0,7793.300518134715,193.0]), label=1),
    Row(userFeatures=Vectors.sparse(658,[0,1,2,3,4,5,6,19,301,503,608,609,610,611,612,613,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,630,631,632,633,634,635,636,637,638,639,640,641,642,643,644,645,646,647,648,649,650,651,652,653,654,655,656,657],[7788.0,6.0,685.0,76.0,12.0,2.0,187.0,1.0,1.0,1.0,2.0,6.0,3.0,9.0,76.0,6.0,685.0,7788.0,76.0,5776.0,438976.0,6.0,456.0,34656.0,36.0,2736.0,216.0,685.0,52060.0,3956560.0,4110.0,312360.0,24660.0,469225.0,3.56611E7,2815350.0,3.21419125E8,7788.0,591888.0,4.4983488E7,46728.0,3551328.0,280368.0,5334780.0,4.0544328E8,3.200868E7,3.6543243E9,6.0652944E7,4.609623744E9,3.63917664E8,4.154726664E10,4.72365127872E11,3044.0,7814.0,6099.099633859992,50254.0,7777.0,7809.0,7793.300518134715,193.0]), label=1)
    ])

test = spark.createDataFrame([
    Row(userFeatures=Vectors.sparse(658,[0,1,2,3,4,5,6,19,301,503,608,609,610,611,612,613,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,630,631,632,633,634,635,636,637,638,639,640,641,642,643,644,645,646,647,648,649,650,651,652,653,654,655,656,657],[7809.0,6.0,815.0,71.0,12.0,2.0,187.0,1.0,1.0,1.0,1.0,6.0,5.0,9.0,71.0,6.0,815.0,7809.0,71.0,5041.0,357911.0,6.0,426.0,30246.0,36.0,2556.0,216.0,815.0,57865.0,4108415.0,4890.0,347190.0,3333.0,33.0,4.7159975E7,3985350.0,5.41343375E8,7809.0,554439.0,3.9365169E7,46854.0,3326634.0,281124.0,6364335.0,4.51867785E8,3.818601E7,5.186933025E9,6.0980481E7,4.329614151E9,3.65882886E8,4.9699092015E10,4.76196576129E11,3044.0,7814.0,6099.099633859992,50254.0,7777.0,7809.0,7793.300518134715,193.0]), label=1),
    Row(userFeatures=Vectors.sparse(658,[0,1,2,3,4,5,6,19,301,503,608,609,610,611,612,613,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,630,631,632,633,634,635,636,637,638,639,640,641,642,643,644,645,646,647,648,649,650,651,652,653,654,655,656,657],[7788.0,6.0,685.0,76.0,12.0,2.0,187.0,1.0,1.0,1.0,2.0,6.0,3.0,9.0,76.0,6.0,685.0,7788.0,76.0,5776.0,438976.0,6.0,456.0,34656.0,36.0,2736.0,216.0,685.0,52060.0,3956560.0,4110.0,312360.0,23423.0,2.0,3.56611E7,2815350.0,3.21419125E8,7788.0,591888.0,4.4983488E7,46728.0,3551328.0,280368.0,5334780.0,4.0544328E8,3.200868E7,3.6543243E9,6.0652944E7,4.609623744E9,3.63917664E8,4.154726664E10,4.72365127872E11,3044.0,7814.0,6099.099633859992,50254.0,7777.0,7809.0,7793.300518134715,193.0]), label=0)
    ])

move = 7 # From budget 
moves = []
possibility = [0,1,2,3,4,5,6,19,301,503,608,609,610,611,612,613,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,630,631,632,633,634,635,636,637,638,639,640,641,642,643,644,645,646,647,648,649,650,651,652,653,654,655,656,657]

while move > 0:
    choice = random.choice(possibility)
    possibility.remove(choice)
    moves.append(choice)
    move-=1
   
col = "_".join([str(el) for el in moves]) 
slicer = VectorSlicer(inputCol="userFeatures", outputCol=f"features", indices=moves)
#df = slicer.transform(df)
#training = df.select(f"index_{col}_features", "label").withColumnRenamed(f"index_{col}_features", "features")


lr = LogisticRegression(maxIter=10)


pipeline = Pipeline(stages=[slicer,lr])

full_model = pipeline.fit(df) 


prediction = full_model.transform(test)

prediction.show()

evaluator = BinaryClassificationEvaluator()

#evaluator.setRawPredictionCol("rawPrediction")

print(evaluator.evaluate(prediction))

#lrModel = lr.fit(training) 

#assembler = VectorAssembler(), concatenate and evaluate 
