import sys
import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.ml.classification import LinearSVC 
from automl_iasd.feature_engineering.util import get_max_count_distinct
from automl_iasd.feature_processing.util import get_numeric_columns_with_null_values, get_categorical_columns_with_null_values, get_categorical_columns, get_numeric_columns
from automl_iasd.feature_processing.cleaning import fill_missing_values, remove_outliers
from automl_iasd.feature_processing.encode import encode_categorical_features
from automl_iasd.feature_processing.scaling import normalize, standardize
from automl_iasd.feature_engineering.transformations import apply_discretization, apply_polynomial_expansion, apply_binary_transformation, apply_group_by_then_transformation, create_features_column
from automl_iasd.feature_engineering.selection import nrpa_feature_selector
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll import scope
from pyspark.ml.feature import VectorSlicer, VectorIndexer, StringIndexer
from pyspark.ml import Pipeline
import logging
import random
import boto3
import json

# Get arguments for process
data_path = sys.argv[1]
budget = sys.argv[2]
task = sys.argv[3]
label_column_name = sys.argv[4]
model_path = sys.argv[5]

# Set up session
spark = SparkSession.builder.getOrCreate()
spark.sparkContext.addPyFile("./automl-iasd-0.1.0.tar.gz")

# Load the dataset
dataframe = spark.read.parquet(f"{data_path}")
full_train_set_dataframe, test_set_dataframe = dataframe.randomSplit([0.8,0.2])

# Need to keep the feature selection in place and search throught the hyperparameters 

def generate_best_feature_subset(full_train_set_dataframe, label_column_name, task, budget):
	classification_algorithms_stages = {
			"DecisionTreeClassifier" : [],
			"LogisticRegression" : [],
			"RandomForestClassifier": [],
			"NaiveBayes" : [],
			"LinearSVC" : []
		}
	classification_algorithms = [ # Could be good to find a better initialization 
				LinearSVC(maxIter=20)
			]

	train_set_dataframe_for_selection = full_train_set_dataframe
	train_set_dataframe_for_selection.cache()
	pipeline_stages = []
	budget = int(budget)

	logging.info("AutoFE : Data preprocessing - Performing eventual missing values imputation ... ")
	numeric_column_with_null_value = get_numeric_columns_with_null_values(train_set_dataframe_for_selection)
	categorical_column_with_null_value = get_categorical_columns_with_null_values(train_set_dataframe_for_selection)
	has_missing_value = False
	if len(numeric_column_with_null_value)>0 or len(categorical_column_with_null_value)>0:
		has_missing_value = True
		train_set_dataframe_for_selection, imputers = fill_missing_values(numeric_column_with_null_value, categorical_column_with_null_value, train_set_dataframe_for_selection, strategy="expand")
		train_set_dataframe_for_selection = dataframe.drop(*numeric_column_with_null_value)
		train_set_dataframe_for_selection = dataframe.drop(*categorical_column_with_null_value)
		pipeline_stages.append(imputers)

	logging.info("AutoFE : Data preprocessing - Encoding categorical features ... ")
	categorical_columns = get_categorical_columns(train_set_dataframe_for_selection)
	numerical_columns = get_numeric_columns(train_set_dataframe_for_selection)
	# Remove the label from the columns processed
	numerical_columns.remove(label_column_name)
	# Recompute the set of usable features
	categorical_columns_to_encode = list(set(categorical_columns) - set(categorical_column_with_null_value))
	numerical_columns_without_null_value = list(set(numerical_columns) - set(numeric_column_with_null_value))
	train_set_dataframe_for_selection, one_hot_encoded_output_cols, encoders = encode_categorical_features(train_set_dataframe_for_selection, categorical_columns_to_encode)
	for encoder in encoders:
		pipeline_stages.append(encoder)
	
	logging.info("AutoFE : Feature generation - Applying unary operators ... ")
	# Unary transformations
	train_set_dataframe_for_selection, discretizer = apply_discretization(train_set_dataframe_for_selection, numerical_columns_without_null_value)
	train_set_dataframe_for_selection, polynomial_assembler, polynomial_expander = apply_polynomial_expansion(train_set_dataframe_for_selection, numerical_columns_without_null_value)
	pipeline_stages.append(discretizer)
	pipeline_stages.append(polynomial_assembler)
	pipeline_stages.append(polynomial_expander)	

	logging.info("AutoFE : Feature generation - Applying binary and group by then operators ... ")
	# Binary transformations
	feature_generation_budget = budget 
	while(feature_generation_budget>0):
		transformation = random.choice(["Binary", "GroupByThen"])
		# arithmetic transformations 
		if transformation == "Binary":
			#Choose two numeric columns 
			columns = random.choices(numerical_columns_without_null_value, k=2)
			train_set_dataframe_for_selection, arithmetic_transformer = apply_binary_transformation(train_set_dataframe_for_selection, columns)
			pipeline_stages.append(arithmetic_transformer)

		else: #GroupByThen
			numeric_column = random.choice(numerical_columns_without_null_value)
			categorical_column = random.choice(categorical_columns_to_encode)
			train_set_dataframe_for_selection, group_by_then_transformer = apply_group_by_then_transformation(train_set_dataframe_for_selection, categorical_column, numeric_column)
			pipeline_stages.append(group_by_then_transformer)
		feature_generation_budget-=1

	logging.info("AutoFE : Feature generation - Creating feature column ... ")
	columns_to_featurized = train_set_dataframe_for_selection.schema.names
	columns_to_featurized = [ele for ele in columns_to_featurized if ele not in [label_column_name,*numeric_column_with_null_value,*categorical_column_with_null_value, *categorical_columns_to_encode]]
	train_set_dataframe_for_selection, feature_assembler = create_features_column(train_set_dataframe_for_selection, columns_to_featurized)
	pipeline_stages.append(feature_assembler)

	logging.info("AutoFE : Data Preprocessing - Applying normalizer and standardizer ... ")
	train_set_dataframe_for_selection, normalizer = normalize(train_set_dataframe_for_selection, numerical_columns_without_null_value)
	train_set_dataframe_for_selection, scaler = standardize(train_set_dataframe_for_selection, numerical_columns_without_null_value)
	pipeline_stages.append(normalizer)
	pipeline_stages.append(scaler)

	# At this point, all features have been generated and we can split the dataset to evaluate the features selection that will constitute the last stage of the pipeline
	train_set_dataframe_for_selection, validation_set_dataframe_for_selection = train_set_dataframe_for_selection.randomSplit([0.75,0.25])

	# Baseline algorithm for evaluation
	algorithm = LinearSVC(maxIter=30)
	algorithm_name = str(algorithm).split("_")[0]
	algorithm.setLabelCol(f"{label_column_name}")

	# Metrics for Monte Carlo search
	selection_initial_uniform_policy = [0.5 for _ in range(2*len(columns_to_featurized))]
	selection_budget = budget
	level = selection_budget
	iterations = selection_budget
	algorithm_fe_performance = []
	benchmark = []
	classification_algorithms_stages[algorithm_name] = pipeline_stages.copy()

	logging.info(f"AutoFE : Feature selection - Selecting the best subset of features for {algorithm_name}... ")
	bestScore, feature_selection = nrpa_feature_selector(level, iterations, train_set_dataframe_for_selection, validation_set_dataframe_for_selection, len(columns_to_featurized), algorithm, selection_initial_uniform_policy)
	feature_selection_indices = [i for i in range(len(feature_selection)) if feature_selection[i]=="Keep"] # The best feature selection sent back by the algorithm
	print(bestScore)
	algorithm_fe_performance.append([feature_selection_indices, bestScore])

	feature_selector = VectorSlicer(inputCol="features", outputCol="selectedFeatures", indices=feature_selection_indices)
	classification_algorithms_stages[algorithm_name].append(feature_selector)
	stages = classification_algorithms_stages[algorithm_name]

	return stages, feature_selection, feature_selection_indices, bestScore


auto_fe_stages, feature_selection, feature_selection_indices, bestScore = generate_best_feature_subset(full_train_set_dataframe, label_column_name, task, budget)
# We divide again the training set for HPO validation
train_set_dataframe_for_hpo, validation_set_dataframe_for_hpo = full_train_set_dataframe.randomSplit([0.7,0.3])

def train_svm(regParam, aggregationDepth):

	# Unlink list
	stages = auto_fe_stages.copy()

	# Define the algorithm
	algorithm = LinearSVC(regParam=regParam, aggregationDepth=aggregationDepth)
	algorithm.setFeaturesCol(f"selectedFeatures")
	algorithm.setLabelCol(f"{label_column_name}")
	stages.append(algorithm)

	pipeline = Pipeline(stages=stages)
	model = pipeline.fit(train_set_dataframe_for_hpo)

	logging.info("AutoML : Evaluating the performance of the model on the HPO val set ... ")
	predictions = model.transform(validation_set_dataframe_for_hpo)
	evaluator = BinaryClassificationEvaluator(metricName='areaUnderROC')
	evaluator.setLabelCol(f"{label_column_name}")
	aucroc = evaluator.evaluate(predictions)
	print(f"The AUC error on the val set with a step for feature selection is : {aucroc}")
	return model, aucroc


def train_with_hyperopt(params):
	""" 
	
	Objective function to minimize with the best params

	:param params: hyperparameters as a dict. Its structure is consistent with how search space is defined. See below.
		:return: dict with fields 'loss' (scalar loss) and 'status' (success/failure status of run)
	"""

	regParam = params["regParam"]
	aggregationDepth = params["aggregationDepth"]

	model, aucroc = train_svm(regParam, aggregationDepth)
	loss = - aucroc
	return {'loss': loss, 'status': STATUS_OK}


space = {
  'regParam': hp.uniform('regParam', 0.0, 0.03*float(budget)),
  'aggregationDepth': scope.int(hp.uniform('aggregationDepth', 2, 6)),
}

algo=tpe.suggest

best_params = fmin(
	fn=train_with_hyperopt,
	space=space,
	algo=algo,
	max_evals=12
)


aggregationDepth = int(best_params["aggregationDepth"])

# Now we retrain the model fully 
best_svm = LinearSVC(regParam=best_params["regParam"], aggregationDepth=aggregationDepth)
best_svm.setFeaturesCol(f"selectedFeatures")
best_svm.setLabelCol(f"{label_column_name}")
best_stages = auto_fe_stages.copy()
best_stages.append(best_svm)
best_pipeline = Pipeline(stages=best_stages)
best_model = best_pipeline.fit(full_train_set_dataframe)

# Evaluate the model 
logging.info("AutoML : Evaluating the performance of the best model on the test set ... ")
predictions = best_model.transform(test_set_dataframe)
evaluator = BinaryClassificationEvaluator(metricName='areaUnderROC')
evaluator.setLabelCol(f"{label_column_name}")
aucroc_on_test = evaluator.evaluate(predictions)

# Send the metrics and model

s3 = boto3.client('s3')
json_object = {"Algorithm" : "SVM",
	"aucroc_on_test": aucroc_on_test
}
s3.put_object(
     Body=json.dumps(json_object),
     Bucket='automl_iasd',
     Key=f'{model_path}/{algorithm_name}_{aucroc_on_test}/metrics'
)
best_model.save(f"{model_path}/SVM_{aucroc_on_test}/model")






