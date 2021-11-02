#Imports base stack
from pyspark.sql.functions import when
from pyspark import keyword_only
from pyspark.ml.pipeline import Transformer
from pyspark.ml.param.shared import HasInputCol, HasInputCols, HasOutputCol, HasOutputCols, Param
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression

#Import optimizer 
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval

#Import data preprocessers 
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import FeatureHasher
from pyspark.ml import Pipeline

#Classification algorithm 
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import MultilayerPerceptronClassifier
import matplotlib.pyplot as plt
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorSlicer, VectorIndexer, StringIndexer
from pyspark.ml.feature import Imputer
from automl_iasd.feature_processing.util import get_most_frequent_feature_value
import pyspark.sql.functions as F
from pyspark.sql import functions as F, types as T
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer
import pyspark.sql.functions as F
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import Normalizer
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import when
from pyspark import keyword_only
from pyspark.ml.pipeline import Transformer
from pyspark.ml.param.shared import HasInputCol, HasInputCols, HasOutputCol, HasOutputCols, Param
import random 
import math
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorSlicer, VectorIndexer
# Filter based selector 
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.feature import RFormula
from pyspark.ml.feature import QuantileDiscretizer, PolynomialExpansion
from pyspark.sql.functions import col
from pyspark.sql.functions import when
from pyspark.ml.feature import VectorAssembler
from automl_iasd.feature_engineering.transformers import ArithmeticTransformer, GroupByThenTransformer
from pyspark.sql.functions import when
from pyspark import keyword_only
from pyspark.ml.pipeline import Transformer
from pyspark.ml.param.shared import HasInputCol, HasInputCols, HasOutputCol, HasOutputCols, Param
from pyspark.sql.functions import countDistinct


def fill_missing_values(numeric_columns_to_fill, categorical_columns_to_fill, dataframe, strategy="expand"):
	"""Fill the missing values by applying all possible transformation to each feature."""
	raw_numeric_features_name = [item[0] for item in dataframe.dtypes if item[1].startswith('double') or item[1].startswith('integer')]
	numerical_fill_strategy = ["mean", "median", "mode"]
	categorical_fill_strategy = ["most_common", "Unknown_filling"]
	transformations = []
	for strategy in numerical_fill_strategy:
		imputed_features_names = [f"imputed_{strategy}_"+feature for feature in numeric_columns_to_fill]
		imputer = Imputer(inputCols=[*numeric_columns_to_fill], outputCols=[*imputed_features_names])
		model = imputer.fit(dataframe)
		dataframe = model.transform(dataframe)
		transformations.append(imputer)

	for strategy in categorical_fill_strategy:
		if strategy == "most_common":
			for column in categorical_columns_to_fill:
				most_common_feature_value = get_most_frequent_feature_value(dataframe, column)
				dataframe = dataframe.withColumn(f"{strategy}_{column}", F.when(F.col(f"{column}").isNull() | F.isnan(F.col(f"{column}")) | F.col(f"{column}").contains('?') | F.col(f"{column}").contains('None')| F.col(f"{column}").contains('Null') | (F.col(f"{column}") == ''),most_common_feature_value).otherwise(dataframe[f"{column}"]))
			transformations.append(imputer)
		if strategy == "Unknown_filling":
			for column in categorical_columns_to_fill:
				dataframe = dataframe.withColumn(f"{strategy}_{column}", F.when(F.col(f"{column}").isNull() | F.isnan(F.col(f"{column}")) | F.col(f"{column}").contains('?') | F.col(f"{column}").contains('None')| F.col(f"{column}").contains('Null') | (F.col(f"{column}") == ''),"Unkown").otherwise(dataframe[f"{column}"]))
			transformations.append(imputer)
	return dataframe, transformations

def remove_outliers(dataframe, numerical_columns):
	"""Remove outliers from the dataframe."""

	# TODO : Compute visualisation 
	

	# TODO :  Send the visualisation to S3 
	
	# Find the outliers 
	scaler = StandardScaler()
	classifier = IsolationForest(contamination="auto", random_state=42, n_jobs=-1)
	x_train = [i[0:len(numerical_columns)] for i in df.select(numerical_columns).collect()]

	SCL = spark_session.sparkContext.broadcast(scaler)
	CLF = spark_session.sparkContext.broadcast(clf)
	
	udf_predict_using_broadcasts = F.udf(predict_using_broadcasts, T.IntegerType())
	
	dataframe = dataframe.withColumn('prediction',
    udf_predict_using_broadcasts())

	# Remove them 


	return dataframe


def encode_categorical_features(dataframe, categorical_features):
	"""Encode categorical features and add to the column of the dataframe."""
	transformations = []

	# Ordinal encoding 
	ordinal_encoded_output_cols = ["ordinal_indexed_"+categorical_feature for categorical_feature in categorical_features]
	indexer = StringIndexer(inputCols=categorical_features, outputCols=ordinal_encoded_output_cols, handleInvalid="keep")
	dataframe = indexer.fit(dataframe).transform(dataframe)

	transformations.append(indexer)

	# One-Hot-Encoding
	one_hot_encoded_output_cols = ["one_hot_encoded_"+categorical_feature for categorical_feature in categorical_features]
	encoder = OneHotEncoder(inputCols=ordinal_encoded_output_cols,
                        outputCols=one_hot_encoded_output_cols)
	model = encoder.fit(dataframe)
	dataframe = model.transform(dataframe)

	transformations.append(encoder)

	return dataframe, one_hot_encoded_output_cols, transformations



def normalize(dataframe, columns):
	"""Normalize the columns from the dataframe."""
	normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=1.0)
	dataframe = normalizer.transform(dataframe)
	return dataframe, normalizer

def standardize(dataframe, columns):
	"""Standardize the columns from the dataframe."""
	scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=False)
	scalerModel = scaler.fit(dataframe)
	dataframe = scalerModel.transform(dataframe)
	return dataframe, scaler



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



class ArithmeticTransformer(Transformer, HasInputCols, HasOutputCols):
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


class GroupByThenTransformer(Transformer, HasInputCols, HasOutputCols):
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






class SelectionState:
	"""docstring for State"""
	def __init__(self, selected_features, not_selected_features, to_select_features, train_set_dataframe, validation_set_dataframe):
		self.selected_features = selected_features
		self.not_selected_features = not_selected_features
		self.to_select_features = to_select_features
		self.next_feature_to_select = to_select_features[0]
		self.train_set_dataframe = train_set_dataframe
		self.validation_set_dataframe = validation_set_dataframe
		self.feature_space_size = len(to_select_features+not_selected_features+selected_features)

	def isTerminal(self):
		return len(self.to_select_features) == 0

	def pick_feature(self, move):
		if move == "Keep":
			self.selected_features.append(self.next_feature_to_select)
		else:
			self.not_selected_features.append(self.next_feature_to_select)

		self.to_select_features.remove(self.next_feature_to_select)
		if len(self.to_select_features) > 0:
			self.next_feature_to_select = self.to_select_features[0]

def nrpa_feature_selector(level, iterations, train_set_dataframe, validation_set_dataframe, feature_space_size, learning_algorithm, policy): 
	"""Take a dataframe with a feature column and return the best subset of feature for the model."""
	root = SelectionState(selected_features=[], not_selected_features=[], to_select_features=list(range(feature_space_size)), train_set_dataframe=train_set_dataframe, validation_set_dataframe=validation_set_dataframe)
	if level == 0:
		return playout(root, policy, learning_algorithm)
	best_score = float('-inf')
	for N in range(iterations):
		result, new = nrpa_feature_selector(level-1, iterations, train_set_dataframe, validation_set_dataframe, feature_space_size, learning_algorithm, policy)
		if result>=best_score:
			best_score = result
			seq = new
		policy = adapt(policy, seq, feature_space_size, train_set_dataframe, validation_set_dataframe)
	return best_score, seq


def snrpa_feature_selector(level, iterations, P, train_set_dataframe, validation_set_dataframe, feature_space_size, learning_algorithm, policy): 
	if level == 0:
		root = SelectionState(selected_features=[], not_selected_features=[], to_select_features=list(range(feature_space_size)), train_set_dataframe=train_set_dataframe, validation_set_dataframe=validation_set_dataframe)
		return playout(root, policy, learning_algorithm)
	elif level == 1:
		best_score = float('-inf')
		for _ in range(P):
			result, new = snrpa_feature_selector(level-1, iterations, P, train_set_dataframe, validation_set_dataframe, feature_space_size, learning_algorithm, policy)
			if result>=best_score:
				best_score = result
				seq = new
		return best_score, seq
	else:
		best_score = float("-inf")
		for N in range(iterations):
			result, new = snrpa_feature_selector(level-1, iterations, P, train_set_dataframe, validation_set_dataframe, feature_space_size, learning_algorithm, policy)
			if result>=best_score:
				best_score = result
				seq = new
			policy = adapt(policy, seq, feature_space_size, train_set_dataframe, validation_set_dataframe)
		return best_score, seq


def playout(state, policy, learning_algorithm):
	sequence = []
	while True:
		if state.isTerminal(): 
			return score(state, learning_algorithm), sequence
		z=0.0
		for m in ["Keep", "Discard"]:
			z=z+math.exp(policy[code(m, state)])
		move = choose_a_move(policy, z, state)
		state = play(state, move)
		sequence.append(move)

def code(move, state): 
	feature_space_size = state.feature_space_size 
	to_select_features = state.to_select_features 
	if move == "Keep":
		move_policy_index = (feature_space_size - len(to_select_features))*2
	else: # "Discard" 
		move_policy_index = ((feature_space_size - len(to_select_features))*2)+1
	return move_policy_index  

def choose_a_move(policy, z, state):
	feature_space_size = state.feature_space_size 
	to_select_features = state.to_select_features 
	probability = [math.exp(policy[code("Keep", state)])/z, math.exp(policy[code("Discard", state)])/z]
	choice = random.choices(["Keep","Discard"], weights=probability, k=1)
	return choice[0]

def play(state, move):
	state.pick_feature(move)
	return state

def adapt(policy, sequence, feature_space_size, train_set_dataframe, validation_set_dataframe):
	polp = policy
	state = SelectionState(selected_features=[], not_selected_features=[], to_select_features=list(range(feature_space_size)), train_set_dataframe=train_set_dataframe, validation_set_dataframe=validation_set_dataframe)
	alpha = 0.5
	for move in sequence:
		polp[code(move, state)] = polp[code(move, state)]+alpha
		z=0.0
		for m in ["Keep", "Discard"]:
			z=z+math.exp(policy[code(m, state)])
		for m in ["Keep", "Discard"]:
			polp[code(m, state)] = polp[code(m, state)]-alpha*math.exp(policy[code(m,state)])/z
		state = play(state, move)
	policy = polp
	return policy

def score(state, learning_algorithm):
	train_set_dataframe = state.train_set_dataframe
	validation_set_dataframe = state.validation_set_dataframe
	selected_features = state.selected_features
	input_features_col = "features"
	learning_algorithm_name = str(learning_algorithm).split("_")[0]

	selector = VectorSlicer(inputCol=input_features_col, outputCol="selectedFeatures", indices=selected_features)
	training_df = selector.transform(train_set_dataframe)
	validation_df = selector.transform(validation_set_dataframe)
	learning_algorithm.setFeaturesCol(f"selectedFeatures")

	if learning_algorithm_name == "DecisionTreeClassifier" or learning_algorithm_name == "RandomForestClassifier":
		featureIndexer = VectorIndexer(inputCol="selectedFeatures", outputCol="indexedFeatures", maxCategories=32)
		fi_tr = featureIndexer.fit(training_df)
		fi_va = featureIndexer.fit(validation_df)
		training_df = fi_tr.transform(training_df)
		validation_df = fi_va.transform(validation_df)
		learning_algorithm.setFeaturesCol("indexedFeatures")

	if learning_algorithm_name == "MultilayerPerceptronClassifier":
		learning_algorithm.setLayers([len(selected_features), len(selected_features), 2])

	model = learning_algorithm.fit(training_df)
	prediction = model.transform(validation_df)

	evaluator = BinaryClassificationEvaluator(metricName='areaUnderROC')
	evaluator.setLabelCol(learning_algorithm.getLabelCol())
	print(evaluator.evaluate(prediction))
	print(selected_features)
	return evaluator.evaluate(prediction)




def apply_discretization(dataframe, columns):
	"""Apply discretization."""
	output_columns = ["discretized_"+col for col in columns]
	discretizer = QuantileDiscretizer(numBuckets=10, inputCols=columns, outputCols=output_columns)
	dataframe = discretizer.fit(dataframe).transform(dataframe)
	return dataframe, discretizer#, output_columns

def apply_polynomial_expansion(dataframe, columns):
	"""Apply polynomial expansion."""
	#output_columns = ["polynomial_expanded_"+col for col in columns]
	output_columns = "_".join(columns)
	polynomial_assembler = VectorAssembler(
	    inputCols=columns,
	    outputCol="vectorized_"+output_columns)

	dataframe = polynomial_assembler.transform(dataframe)

	polyExpansion = PolynomialExpansion(degree=3, inputCol="vectorized_"+output_columns, outputCol="feature_polynomially_expanded_"+output_columns)
	dataframe = polyExpansion.transform(dataframe)
	return dataframe, polynomial_assembler, polyExpansion#, output_columns


def apply_binary_transformation(dataframe, columns):
	"""Apply all binary transformations on columns passed."""

	column_name = "_".join(columns)
	column_name_reverse = "_".join(columns[::-1])

	outputCols = "_".join(columns)

	arithmetic_transformer = ArithmeticTransformer(inputCols=columns, outputCols=[f"sum_{column_name}", f"difference_{column_name}", f"difference_{column_name_reverse}", f"multiplied_{column_name}", f"division_{column_name}", f"division_{column_name_reverse}"])

	dataframe = arithmetic_transformer.transform(dataframe)

	return dataframe, arithmetic_transformer


def apply_group_by_then_transformation(dataframe, categorical_column, numeric_column):
	"""Apply all unary transformations on columns passed."""

	group_by_then_transformer = GroupByThenTransformer(inputCols=[categorical_column, numeric_column], outputCols=["min", "max", "avg", "count"])
	dataframe = group_by_then_transformer.transform(dataframe)

	return dataframe, group_by_then_transformer


def create_features_column(dataframe, columns):
	"""Create a feature column from the dataframe."""
	assembler = VectorAssembler(
	    inputCols=columns,
	    outputCol="features")

	dataframe = assembler.transform(dataframe)
	return dataframe, assembler



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


def get_max_count_distinct(dataframe, columns):
	max_count_distinct = 0
	for column in columns:
		count_distinct = dataframe.select(countDistinct(f"{column}")).head()[0]
		if count_distinct>max_count_distinct:
			max_count_distinct = count_distinct
	return max_count_distinct+1
