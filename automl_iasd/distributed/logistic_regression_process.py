# LogisticRegressionProcess
import sys
#import findspark
#findspark.init()
from pyspark.sql import SparkSession

# Get arguments for process
data_path = sys.argv[1]
budget = sys.argv[2]
task = sys.argv[3]
label_column_name = sys.argv[4]

# Set up session
spark = SparkSession.builder.getOrCreate()
#spark.sparkContext.addPyFile("./automl-iasd-0.1.0.tar.gz")

# Get the rest of the dependencies 

# To resolve 
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK



# Load the dataset

dataframe = spark.read.parquet("s3://automl-iasd/airlines/dataset/airlines.parquet/")
full_train_set_dataframe, test_set_dataframe = dataframe.randomSplit([0.8,0.2])


def train_logistic_regression(regParam=0.0, elasticNetParam=0.0):
	# Object properties 
	classification_algorithms_stages = {
			"DecisionTreeClassifier" : [],
			"LogisticRegression" : [],
			"RandomForestClassifier": [],
			"NaiveBayes" : [],
			"MultilayerPerceptronClassifier" : []
		}
	classification_algorithms = [ # Could be good to find a better initialization 
				MultilayerPerceptronClassifier(maxIter=20),
				DecisionTreeClassifier(maxDepth=2),
				LogisticRegression(maxIter=20), 
				RandomForestClassifier(maxDepth=2),
				NaiveBayes()
			]

	print("-------------------------------------------------------------")
	print("---------------Starting the run of the pipeline--------------")
	print("-------------------------------------------------------------")
	print("The initial dataframe is the one below :")
	logging.getLogger().setLevel(logging.INFO)
	pipeline_stages = []


	logging.info("AutoFE - Splitting dataset ... ")
	train_set_dataframe, validation_set_dataframe_for_hyperparams = full_train_set_dataframe.randomSplit([0.8,0.2])
	train_set_dataframe_for_selection = train_set_dataframe

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
	columns_to_featurized = dataframe.schema.names
	columns_to_featurized = [ele for ele in columns_to_featurized if ele not in [label_column_name,*numeric_column_with_null_value,*categorical_column_with_null_value, *categorical_columns_to_encode]]
	train_set_dataframe_for_selection, feature_assembler = create_features_column(train_set_dataframe_for_selection, columns_to_featurized)
	pipeline_stages.append(feature_assembler)

	logging.info("AutoFE : Data Preprocessing - Applying normalizer and standardizer ... ")
	train_set_dataframe_for_selection, normalizer = normalize(train_set_dataframe_for_selection, numerical_columns_without_null_value)
	train_set_dataframe_for_selection, scaler = standardize(train_set_dataframe_for_selection, numerical_columns_without_null_value)
	pipeline_stages.append(normalizer)
	pipeline_stages.append(scaler)

	train_set_dataframe_for_selection, validation_set_dataframe_for_selection, = train_set_dataframe_for_selection.randomSplit([0.66,0.34])
	selection_initial_uniform_policy = [0.5 for _ in range(2*len(columns_to_featurized))]
	algorithm = LogisticRegression(regParam=regParam, elasticNetParam=elasticNetParam)
	algorithm_name = str(algorithm).split("_")[0]
	algorithm.setLabelCol(f"{slabel_column_name}")


	selection_budget = budget
	level = selection_budget
	iterations = selection_budget
	algorithm_fe_performance = []
	benchmark = []

	# The case of "DecisionTreeClassifier" and "RandomForestClassifier" to uncomment for thoses files
	# labelIndexer = StringIndexer(inputCol=f"{label_column_name}", outputCol="indexedLabel")
	# li_tr = labelIndexer.fit(train_set_dataframe)
	# li_va = labelIndexer.fit(validation_set_dataframe)
	# train_set_dataframe = li_tr.transform(train_set_dataframe) 
	# validation_set_dataframe = li_va.transform(validation_set_dataframe)

	classification_algorithms_stages[algorithm_name] = pipeline_stages.copy()

	# The case of "DecisionTreeClassifier" and "RandomForestClassifier" to uncomment for thoses files
	# if algorithm_name == "DecisionTreeClassifier" or algorithm_name == "RandomForestClassifier":
	# 	classification_algorithms_stages[algorithm_name].append(labelIndexer)
	# 	algorithm.setLabelCol("indexedLabel")
	# 	maximum_number_of_categories = get_max_count_distinct(train_set_dataframe, categorical_columns_to_encode)
	# 	algorithm.setMaxBins(maximum_number_of_categories)


	logging.info(f"AutoFE : Feature selection - Selecting the best subset of features for {algorithm_name}... ")
	bestScore, feature_selection = nrpa_feature_selector(level, iterations, train_set_dataframe_for_selection, validation_set_dataframe_for_selection, len(columns_to_featurized), algorithm, selection_initial_uniform_policy)
	feature_selection_indices = [i for i in range(len(feature_selection)) if feature_selection[i]=="Keep"]
	print(bestScore)
	algorithm_fe_performance.append([algorithm_name, feature_selection_indices, bestScore])

	feature_selector = VectorSlicer(inputCol="features", outputCol="selectedFeatures", indices=feature_selection_indices)
	classification_algorithms_stages[algorithm_name].append(feature_selector)

	# The case of "DecisionTreeClassifier" and "RandomForestClassifier" to uncomment for thoses files
	# if algorithm_name == "DecisionTreeClassifier" or algorithm_name == "RandomForestClassifier":
	# 	featureIndexer = VectorIndexer(inputCol="selectedFeatures", outputCol="indexedFeatures", maxCategories=32)
	# 	classification_algorithms_stages[algorithm_name].append(featureIndexer)

	algorithm.setFeaturesCol(f"selectedFeatures")
	classification_algorithms_stages[algorithm_name].append(algorithm)

	pipeline = Pipeline(stages=classification_algorithms_stages[algorithm_name])
	model = pipeline.fit(train_set_dataframe)

	logging.info("AutoFE : Evaluating the performance of the model on the hpo val set ... ")
	#model.setLabelCol(f"{self.label_column_name}")
	predictions = model.transform(validation_set_dataframe_for_hyperparams)
	evaluator = BinaryClassificationEvaluator(metricName='areaUnderROC')
	evaluator.setLabelCol(f"{label_column_name}")

	aucroc = evaluator.evaluate(predictions)
	print(f"The AUC error on the val set is : {aucroc}")

	return model, aucroc


model, aucroc = train_logistic_regression()


# def train_with_hyperopt(params):
# 		""" 
	
# 	Objective function to minimize with the best params

# 	:param params: hyperparameters as a dict. Its structure is consistent with how search space is defined. See below.
# 		:return: dict with fields 'loss' (scalar loss) and 'status' (success/failure status of run)
# 	"""
# 	regParam = params["regParam"]
# 	elasticNetParam = params["elasticNetParam"]
# 	model, aucroc = run(regParam, elasticNetParam)
# 	loss = - aucroc
# 	return {'loss': loss, 'status': STATUS_OK}


