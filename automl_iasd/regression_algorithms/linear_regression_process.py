import sys
import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression 
from pyspark.ml.evaluation import RegressionEvaluator
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll import scope
from pyspark.ml.feature import VectorSlicer, VectorIndexer, StringIndexer
from pyspark.ml import Pipeline
import logging
import random
import boto3
import json
import sagemaker
import mlflow
from sagemaker.session import Session
from time import gmtime, strftime, sleep
from pyspark.sql.functions import  monotonically_increasing_id
from pyspark.sql.functions import lit
from datetime import datetime
import time 
from pyspark.sql.functions import kurtosis, skewness
from pyspark.ml.stat import Correlation
import os
import sagemaker
from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.feature_definition import FeatureDefinition, FeatureTypeEnum

spark = SparkSession.builder.config('spark.hadoop.fs.s3a.access.key', AWS_ACCESS_KEY).config('spark.hadoop.fs.s3a.secret.key', AWS_SECRET_KEY).getOrCreate()
spark.sparkContext.addPyFile("../dist/automl-iasd-0.1.0.tar")
spark.sparkContext.setLogLevel("ERROR")

from automl_iasd.feature_engineering.util import get_max_count_distinct
from automl_iasd.data_preprocessing.util import get_numerical_columns_with_missing_values, get_categorical_columns_with_missing_values, get_categorical_columns, get_numerical_columns
from automl_iasd.data_preprocessing.cleaning import fill_missing_values, remove_outliers
from automl_iasd.data_preprocessing.encode import encode_categorical_features
from automl_iasd.data_preprocessing.scaling  import normalize, standardize
from automl_iasd.feature_engineering.transformations  import apply_discretization, apply_polynomial_expansion, apply_binary_transformation, apply_group_by_then_transformation, create_features_column
from automl_iasd.feature_engineering.selection  import nrpa_feature_selector



# Get arguments for process
dataset = sys.argv[1]
label_column_name = sys.argv[2]
task = sys.argv[3]
metric = sys.argv[4]
budget = sys.argv[5]
automl_instance_model_path = sys.argv[6]
train_only = sys.argv[7]
bucket_name = sys.argv[8]
iam_role = sys.argv[9]
AWS_ACCESS_KEY = sys.argv[10]
AWS_SECRET_KEY = sys.argv[11]

# Set metric
if metric == "None":
	metric = None

if train_only == "True":
	data_path = f"s3a://{bucket_name}/{dataset}/dataset/{dataset}-train.parquet/"
elif train_only == "False":
	data_path = f"s3a://{bucket_name}/{dataset}/dataset/{dataset}.parquet/"
else:
	raise ValueError

dataframe = spark.read.parquet(data_path)
logging.getLogger().setLevel(logging.INFO)

algorithm_name = "LinearRegression"

# We start by splitting the dataset, one part will be used to validate the selection of features and the hyperparameters, the other one will be used to test at the end 
full_train_set_dataframe, test_set_dataframe = dataframe.randomSplit([0.8,0.2])

# Need to keep the feature selection in place and search throught the hyperparameters 

def generate_best_feature_subset(full_train_set_dataframe, label_column_name, task, budget):
	""" """
	classification_algorithms_stages = {
			"DecisionTreeClassifier" : [],
			"linearRegression" : [],
			"RandomForestClassifier": [],
			"NaiveBayes" : [],
			"MultilayerPerceptronClassifier" : []
		}
	classification_algorithms = [ # Could be good to find a better initialization 
				LinearRegression(maxIter=20)
			]

	# Start by making a copy the dataframe so we keep an original for the last training 
	train_set_dataframe_for_selection = full_train_set_dataframe
	train_set_dataframe_for_selection.cache()
	pipeline_stages = []
	budget = int(budget)
	initial_numerical_columns = get_numerical_columns(train_set_dataframe_for_selection)

	# Seek the columns with null values
	logging.info("AutoFE : Data preprocessing - Performing eventual missing values imputation ... ")
	numerical_columns_with_missing_values = get_numerical_columns_with_missing_values(train_set_dataframe_for_selection)
	categorical_columns_with_missing_values = get_categorical_columns_with_missing_values(train_set_dataframe_for_selection)

	# If some columns have missing values, we add corresponding imputed columns with the 3 defined strategy for that column
	has_missing_value = False
	if len(numerical_columns_with_missing_values)>0 or len(categorical_columns_with_missing_values)>0:
		has_missing_value = True
		train_set_dataframe_for_selection, imputers = fill_missing_values(numerical_columns_with_missing_values, categorical_columns_with_missing_values, train_set_dataframe_for_selection, strategy="expand")
		train_set_dataframe_for_selection = dataframe.drop(*numerical_columns_with_missing_values)
		train_set_dataframe_for_selection = dataframe.drop(*categorical_columns_with_missing_values)
		print(f"imputers are : {imputers}")
		for imputer in imputers:
			pipeline_stages.append(imputer)

	print(pipeline_stages)

	# We don't want to process the label column, so we check if the label column is a string to see if it needs to be removed from the numerical or categorical columns
	label_column_is_string = False
	for column in full_train_set_dataframe.dtypes:
		if column[0] == label_column_name and column[1] == 'string':
			label_column_is_string = True

	# We now get the numerical columns which do not have missing values so they can be reduced via PCA, outliers can be removed and they can be printed
	numerical_columns_for_outliers = list(set(initial_numerical_columns) - set(numerical_columns_with_missing_values))

	logging.info("AutoFE : Data preprocessing - Removing outliers and saving visualizations ... ")
	train_set_dataframe_for_selection = remove_outliers(train_set_dataframe_for_selection, numerical_columns_for_outliers, bucket_name, label_column_name, dataset)

	# We now get the categorical columns to encode them with the strategies defined
	logging.info("AutoFE : Data preprocessing - Encoding categorical features ... ")
	categorical_columns = get_categorical_columns(train_set_dataframe_for_selection)
	# We retrieve the numerical columns again which now could potentially contain imputed columns (probably be useless)
	numerical_columns = get_numerical_columns(train_set_dataframe_for_selection)
	# We remove the label column from the columns that will be processed
	if label_column_is_string:
		categorical_columns.remove(label_column_name)
	else:
		numerical_columns.remove(label_column_name)

	# We now fix again the set of numerical features that we will use for feature engineering operations 
	# (we discard the ones with missing values here and do it now since we don't want to include columns of categorical encodings)
	numerical_columns_without_missing_values = list(set(numerical_columns) - set(numerical_columns_with_missing_values))

	# We get the set of categorial columns to encode, i.e. the ones which do not have missing values and we add columns with all strategies
	categorical_columns_to_encode = list(set(categorical_columns) - set(categorical_columns_with_missing_values))
	train_set_dataframe_for_selection, one_hot_encoded_output_cols, encoders = encode_categorical_features(train_set_dataframe_for_selection, categorical_columns_to_encode)
	for encoder in encoders:
		pipeline_stages.append(encoder) # The encoders encode the categorical feature in one vector column

	
	# We start by applying the defined unary operations (discretization and polynomial expansion for numerical features) on all usable numerical columns
	logging.info("AutoFE : Feature generation - Applying unary operators ... ")
	# Unary transformations
	train_set_dataframe_for_selection, discretizer = apply_discretization(train_set_dataframe_for_selection, numerical_columns_without_missing_values)
	train_set_dataframe_for_selection, polynomial_assembler, polynomial_expander, vectorized_features_for_expansion = apply_polynomial_expansion(train_set_dataframe_for_selection, numerical_columns_without_missing_values)
	pipeline_stages.append(discretizer)
	# We add a polynomial assembler cause the features need to be put in a vector column to be polynomially expanded
	pipeline_stages.append(polynomial_assembler)
	pipeline_stages.append(polynomial_expander)	

	# We now apply the binary and group by then operations
	logging.info("AutoFE : Feature generation - Applying binary and group by then operators ... ")

	# Binary transformations

	# We start by setting the budget which is the number of times we will apply an operation
	feature_generation_budget = budget 
	while(feature_generation_budget>0):
		transformation = random.choice(["Binary", "GroupByThen"])
		# arithmetic transformations 
		if transformation == "Binary":
			# We randomly choose two numerical columns 
			columns = random.choices(numerical_columns_without_missing_values, k=2)
			train_set_dataframe_for_selection, arithmetic_transformer = apply_binary_transformation(train_set_dataframe_for_selection, columns)
			pipeline_stages.append(arithmetic_transformer)

		else: #GroupByThen
			# We choose a categorical column (not encoded) to do the group by and a numerical column to make the aggregation 
			numerical_column = random.choice(numerical_columns_without_missing_values)
			categorical_column = random.choice(categorical_columns_to_encode)
			train_set_dataframe_for_selection, group_by_then_transformer = apply_group_by_then_transformation(train_set_dataframe_for_selection, categorical_column, numerical_column)
			pipeline_stages.append(group_by_then_transformer)
		feature_generation_budget-=1

	# We can now create a feature column with all usable columns that will be used for the feature selection process
	logging.info("AutoFE : Feature generation - Creating feature column ... ")
	# We can work on train_set_dataframe_for_selection since it went through all transformations and get a subset of beneficial columns from all the columns added
	all_columns = train_set_dataframe_for_selection.schema.names
	columns_to_featurized = [column for column in all_columns if column not in [label_column_name, vectorized_features_for_expansion, *numerical_columns_with_missing_values, *categorical_columns_with_missing_values, *categorical_columns_to_encode, "initial_numerical_features"]]
	train_set_dataframe_for_selection, feature_assembler = create_features_column(train_set_dataframe_for_selection, columns_to_featurized)
	pipeline_stages.append(feature_assembler)

	print(columns_to_featurized)

	# We can use the feature column created to get the normalized features and the standardized features
	logging.info("AutoFE : Data Preprocessing - Applying normalizer and standardizer ... ")
	train_set_dataframe_for_selection, normalizer = normalize(train_set_dataframe_for_selection)
	train_set_dataframe_for_selection, scaler = standardize(train_set_dataframe_for_selection)
	pipeline_stages.append(normalizer)
	pipeline_stages.append(scaler)

	# At this point, all features have been generated and we can split the dataset to evaluate the features selection that will constitute the last stage of the pipeline
	train_set_dataframe_for_selection, validation_set_dataframe_for_selection = train_set_dataframe_for_selection.randomSplit([0.75,0.25])

	# We start by setting a baseline algorithm that will be used for the selections wrapper evaluation 
	algorithm = LinearRegression(maxIter=30)
	algorithm_name = str(algorithm).split("_")[0]
	algorithm.setLabelCol(f"{label_column_name}")

	# We set the metrics for Monte-Carlo search
	selection_initial_uniform_policy = [0.5 for _ in range(2*len(columns_to_featurized))]
	selection_budget = budget
	level = selection_budget
	iterations = selection_budget
	algorithm_fe_performance = []
	benchmark = []
	feature_space_size = len(columns_to_featurized)
	classification_algorithms_stages[algorithm_name] = pipeline_stages.copy()

	# We now seek the best subset of features
	logging.info(f"AutoFE : Feature selection - Selecting the best subset of features for {algorithm_name}... ")
	bestScore, feature_selection = nrpa_feature_selector(level, iterations, train_set_dataframe_for_selection, validation_set_dataframe_for_selection, feature_space_size, algorithm, selection_initial_uniform_policy, task, metric)
	feature_selection_indices = [i for i in range(len(feature_selection)) if feature_selection[i]=="Keep"] # The best feature selection sent back by the algorithm
	print(bestScore)
	algorithm_fe_performance.append([feature_selection_indices, bestScore])

	feature_selector = VectorSlicer(inputCol="features", outputCol="selectedFeatures", indices=feature_selection_indices)
	classification_algorithms_stages[algorithm_name].append(feature_selector)
	stages = classification_algorithms_stages[algorithm_name]

	train_set_dataframe_for_selection = feature_selector.transform(train_set_dataframe_for_selection)

	return stages, feature_selection, feature_selection_indices, bestScore, train_set_dataframe_for_selection, columns_to_featurized


auto_fe_stages, feature_selection, feature_selection_indices, bestScore, full_train_set_dataframe_with_selection, columns_to_featurized = generate_best_feature_subset(full_train_set_dataframe, label_column_name, task, budget)

print(f"stages are : {auto_fe_stages}")

logging.info(f"The best accuracy found after the automated feature engineering stage is : {bestScore}")


feature_selected_list = [columns_to_featurized[i] for i in feature_selection_indices]
feature_selected = ",".join(feature_selected_list)

logging.info(f"The selected features are : {feature_selected}")

# Save features to feature store 

logging.info("AutoML : Feature store - Creating feature group for the selection ... ")

boto_session = boto3.Session(region_name="eu-west-1")
#role = "arn:aws:iam::792946780047:role/automl_features_store"
role = iam_role 
sagemaker_session = sagemaker.Session()
region = sagemaker_session.boto_region_name
default_bucket = sagemaker_session.default_bucket()
offline_feature_store_bucket = f"s3://{bucket_name}-sagemaker-feature-store"

sagemaker_client = boto_session.client(service_name='sagemaker', region_name=region)

featurestore_runtime = boto_session.client(service_name='sagemaker-featurestore-runtime', region_name=region)

feature_store_session = Session(
    boto_session=boto_session,
    sagemaker_client=sagemaker_client,
    sagemaker_featurestore_runtime_client=featurestore_runtime
)

feature_group_name = f"{dataset}-{algorithm_name}-feature-group-" + strftime('%d-%H-%M-%S', gmtime()) 


timestamp = float(round(time.time()))

full_train_set_dataframe_with_selection = full_train_set_dataframe_with_selection.withColumn(f"{dataset}ID",  monotonically_increasing_id())
full_train_set_dataframe_with_selection = full_train_set_dataframe_with_selection.withColumn(f"EventTime", lit(timestamp))

features_to_define = [feature for feature in full_train_set_dataframe_with_selection.dtypes if feature[0] in feature_selected_list]

# Scale feature names

for i in range(len(features_to_define)):
	if len(features_to_define[i][0])>63:
		features_to_define[i] = (features_to_define[i][0][0:63],features_to_define[i][1])

feature_definitions = [FeatureDefinition(x[0],FeatureTypeEnum("Fractional")) if x[1] in ["double","float"] else FeatureDefinition(x[0],FeatureTypeEnum("Integral")) if x[1] in ["int"] else FeatureDefinition(x[0],FeatureTypeEnum("String")) for x in features_to_define]




feature_definitions.append(FeatureDefinition(f"{dataset}ID", FeatureTypeEnum("String")))
feature_definitions.append(FeatureDefinition(f"EventTime", FeatureTypeEnum("String")))

print(feature_definitions)

feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session, feature_definitions=feature_definitions)

record_identifier_name = f"{dataset}ID"
event_time_feature_name = "EventTime"
now = datetime.now()

# create a FeatureGroup
feature_group.create(
    description = f"{dataset}'s feature group stored for the run on {now}",
    record_identifier_name = record_identifier_name,
    event_time_feature_name = event_time_feature_name,
    role_arn = role,
    s3_uri = offline_feature_store_bucket,
    enable_online_store = True,
    online_store_kms_key_id = None,
    offline_store_kms_key_id = None,
    disable_glue_table_creation = False,
    data_catalog_config = None)


status = feature_group.describe().get("FeatureGroupStatus")
logging.info(f"AutoML : Feature store - Creation of feature group in {offline_feature_store_bucket} launched with status {status} ... ")

# We divide again the training set that we saved in the beginning but for HPO validation this time 
train_set_dataframe_for_hpo, validation_set_dataframe_for_hpo = full_train_set_dataframe.randomSplit([0.7,0.3])

def train_linear_regression(regParam, elasticNetParam):
	""" """
	# Unlink list
	stages = auto_fe_stages.copy()

	# Define the algorithm
	algorithm = LinearRegression(regParam=regParam, elasticNetParam=elasticNetParam)
	algorithm.setFeaturesCol(f"selectedFeatures")
	algorithm.setLabelCol(f"{label_column_name}")
	stages.append(algorithm)

	pipeline = Pipeline(stages=stages)
	model = pipeline.fit(train_set_dataframe_for_hpo)

	logging.info("AutoML : Evaluating the performance of the model on the HPO val set ... ")
	predictions = model.transform(validation_set_dataframe_for_hpo)
	if metric:
		evaluator = RegressionEvaluator(metricName=metric)
	else:
		evaluator = RegressionEvaluator()
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
	elasticNetParam = params["elasticNetParam"]

	model, aucroc = train_linear_regression(regParam, elasticNetParam)
	loss = aucroc
	return {'loss': loss, 'status': STATUS_OK}


space = {
  'regParam': hp.uniform('regParam', 0.0, 0.03*float(budget)),
  'elasticNetParam': hp.uniform('elasticNetParam', 0.0, 0.02*float(budget)),
}

algo=tpe.suggest

logging.info(f"AutoML : Hyperparameters optimization - Seeking the best subset of features with Tree Parzen Estimator ... ")

best_params = fmin(
	fn=train_with_hyperopt,
	space=space,
	algo=algo,
	max_evals=12
)

# Get the numerical columns that will be used for computing the metafeatures 
numerical_columns = get_numerical_columns(full_train_set_dataframe)

## Save metafeatures 
with mlflow.start_run():
	logging.info("AutoML : Meta-features - Computing Kurtosis and skewness for numerical features ... ")
	for i, numerical_feature in enumerate(numerical_columns):
		# Compute asymetrie coefficient
		mlflow.log_metric(f"{numerical_feature}_skewness", full_train_set_dataframe_with_selection.agg(skewness(full_train_set_dataframe_with_selection[numerical_feature])).collect()[0][0])
		# Compute asymetrie coefficient
		mlflow.log_metric(f"{numerical_feature}_kurtosis", full_train_set_dataframe_with_selection.agg(kurtosis(full_train_set_dataframe_with_selection[numerical_feature])).collect()[0][0])
		# Compute asymetrie coefficient
	logging.info("AutoML : Meta-features - Computing features correlation ... ")
	r1 = Correlation.corr(full_train_set_dataframe_with_selection, "selectedFeatures").head()
	mlflow.log_param(f"{dataset}_Pearson_correlation_matrix", str(r1[0]))
	r2 = Correlation.corr(full_train_set_dataframe_with_selection, "selectedFeatures", "spearman").head()
	mlflow.log_param(f"{dataset}_Spearman_correlation_matrix", str(r2[0]))

# TODO : logging.info("AutoML : meta-features - Saving metafeatures to S3 ... ") 

# Now we retrain the model fully 
best_linear_regression = LinearRegression(regParam=best_params["regParam"], elasticNetParam=best_params["elasticNetParam"])
best_linear_regression.setFeaturesCol(f"selectedFeatures")
best_linear_regression.setLabelCol(f"{label_column_name}")
best_stages = auto_fe_stages.copy()
best_stages.append(best_linear_regression)
best_pipeline = Pipeline(stages=best_stages)
best_model = best_pipeline.fit(full_train_set_dataframe)

# Evaluate the model 
logging.info("AutoML : Evaluating the performance of the best model on the test set ... ")
predictions = best_model.transform(test_set_dataframe)
if metric:
	evaluator = RegressionEvaluator(metricName=metric)
else:
	evaluator = RegressionEvaluator()
evaluator.setLabelCol(f"{label_column_name}")
aucroc_on_test = evaluator.evaluate(predictions)

logging.info(f"The accuracy on the test set is : {aucroc_on_test}")

# Send the metrics and model
s3 = boto3.client('s3')
json_object = {"Algorithm" : "LinearRegression",
	"aucroc_on_test": aucroc_on_test
}
# We put the metrics found in a file
s3.put_object(
     Body=json.dumps(json_object),
     Bucket=f'{bucket_name}',
     Key=f'{automl_instance_model_path}/LinearRegression_{aucroc_on_test}/metrics'
)
# We finally end up by saving the found model 
best_model.save(f"s3a://{bucket_name}/{automl_instance_model_path}/LinearRegression_{aucroc_on_test}/model")






