#Imports base stack
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
from automl_iasd.feature_engineering.util import get_max_count_distinct
from automl_iasd.feature_processing.util import get_numeric_columns_with_null_values, get_categorical_columns_with_null_values, get_categorical_columns, get_numeric_columns
from automl_iasd.feature_processing.cleaning import fill_missing_values, remove_outliers
from automl_iasd.feature_processing.encode import encode_categorical_features
from automl_iasd.feature_processing.scaling import normalize, standardize
from automl_iasd.feature_engineering.transformations import apply_discretization, apply_polynomial_expansion, apply_binary_transformation, apply_group_by_then_transformation, create_features_column
from automl_iasd.feature_engineering.selection import nrpa_feature_selector
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorSlicer, VectorIndexer, StringIndexer

#Feature 
from pyspark.ml.feature import Imputer
import logging
import random
import mlflow # TODO
#import featurestore # TODO 

# Visualisation
import matplotlib.pyplot as plt
import numpy as np
 
# Tuning 
from automl_iasd.hyperparameter_optimization.tuning import get_tuned_algorithm


# Got to run for one algorithm only given by the user
class IASDAutoML:
	"""The central class containing the pipeline."""
	def __init__(self, budget, dataframe, label_column_name, task="classification", training_only=False):
		self.budget = budget
		# For large dataframe, just give dataframe path 
		self.dataframe = dataframe
		self.label_column_name = label_column_name
		self.task = task
		self.classification_algorithms = [ # Could be good to find a better initialization 
				MultilayerPerceptronClassifier(maxIter=20),
				DecisionTreeClassifier(maxDepth=2),
				LogisticRegression(maxIter=20), 
				RandomForestClassifier(maxDepth=2),
				NaiveBayes()
			]
		self.regression_algorithms = []
		self.classification_algorithms_stages = {
			"DecisionTreeClassifier" : [],
			"LogisticRegression" : [],
			"RandomForestClassifier": [],
			"NaiveBayes" : [],
			"MultilayerPerceptronClassifier" : []
		}
		
	def run(self, regParam=0.0, elasticNetParam=0.0):
		"""Run the AutoML pipeline.""" 
		print("-------------------------------------------------------------")
		print("---------------Starting the run of the pipeline--------------")
		print("-------------------------------------------------------------")
		print("The initial dataframe is the one below :")
		logging.getLogger().setLevel(logging.INFO)
		dataframe = self.dataframe
		dataframe.show() 
		pipeline_stages = []

		logging.info("AutoFE - Splitting dataset ... ")
		dataframe, test_set_dataframe = dataframe.randomSplit([0.8,0.2])
		# dataframe will undergo all the transformations in order to perform the feature selection algorithm
		# but we want to keep the train set as it is to completely retrain the model with the right hyperparameters
		# at the end. thus, the train_dataframe = dataframe 
		train_dataframe = dataframe


		logging.info("AutoFE : Data preprocessing - Performing eventual missing values imputation ... ")
		numeric_column_with_null_value = get_numeric_columns_with_null_values(self.dataframe)
		categorical_column_with_null_value = get_categorical_columns_with_null_values(self.dataframe)
		has_missing_value = False
		if len(numeric_column_with_null_value)>0 or len(categorical_column_with_null_value)>0:
			has_missing_value = True
			dataframe, imputers = fill_missing_values(numeric_column_with_null_value, categorical_column_with_null_value, self.dataframe, strategy="expand")
			dataframe = dataframe.drop(*numeric_column_with_null_value)
			dataframe = dataframe.drop(*categorical_column_with_null_value)
			pipeline_stages.append(imputers)

		logging.info("AutoFE : Data preprocessing - Encoding categorical features ... ")
		categorical_columns = get_categorical_columns(dataframe)
		numerical_columns = get_numeric_columns(dataframe)
		# Remove the label from the columns processed
		numerical_columns.remove(self.label_column_name)
		# Recompute the set of usable features
		categorical_columns_to_encode = list(set(categorical_columns) - set(categorical_column_with_null_value))
		numerical_columns_without_null_value = list(set(numerical_columns) - set(numeric_column_with_null_value))
		dataframe, one_hot_encoded_output_cols, encoders = encode_categorical_features(dataframe, categorical_columns_to_encode)
		for encoder in encoders:
			pipeline_stages.append(encoder)
		
		logging.info("AutoFE : Feature generation - Applying unary operators ... ")
		# Unary transformations
		dataframe, discretizer = apply_discretization(dataframe, numerical_columns_without_null_value)
		dataframe, polynomial_assembler, polynomial_expander = apply_polynomial_expansion(dataframe, numerical_columns_without_null_value)
		pipeline_stages.append(discretizer)
		pipeline_stages.append(polynomial_assembler)
		pipeline_stages.append(polynomial_expander)	


		logging.info("AutoFE : Feature generation - Applying binary and group by then operators ... ")
		# Binary transformations
		feature_generation_budget = self.budget 
		while(feature_generation_budget>0):
			transformation = random.choice(["Binary", "GroupByThen"])
			# arithmetic transformations 
			if transformation == "Binary":
				#Choose two numeric columns 
				columns = random.choices(numerical_columns_without_null_value, k=2)
				dataframe, arithmetic_transformer = apply_binary_transformation(dataframe, columns)
				pipeline_stages.append(arithmetic_transformer)

			else: #GroupByThen
				numeric_column = random.choice(numerical_columns_without_null_value)
				categorical_column = random.choice(categorical_columns_to_encode)
				dataframe, group_by_then_transformer = apply_group_by_then_transformation(dataframe, categorical_column, numeric_column)
				pipeline_stages.append(group_by_then_transformer)
			feature_generation_budget-=1

		logging.info("AutoFE : Feature generation - Creating feature column ... ")
		columns_to_featurized = dataframe.schema.names
		columns_to_featurized = [ele for ele in columns_to_featurized if ele not in [self.label_column_name,*numeric_column_with_null_value,*categorical_column_with_null_value, *categorical_columns_to_encode]]
		dataframe, feature_assembler = create_features_column(dataframe, columns_to_featurized)
		pipeline_stages.append(feature_assembler)

		logging.info("AutoFE : Data Preprocessing - Applying normalizer and standardizer ... ")
		dataframe, normalizer = normalize(dataframe, numerical_columns_without_null_value)
		dataframe, scaler = standardize(dataframe, numerical_columns_without_null_value)
		pipeline_stages.append(normalizer)
		pipeline_stages.append(scaler)

		train_set_dataframe, validation_set_dataframe, = dataframe.randomSplit([0.66,0.34])
		selection_initial_uniform_policy = [0.5 for _ in range(2*len(columns_to_featurized))]

		selection_budget = self.budget
		level = selection_budget
		iterations = selection_budget
		algorithm_fe_performance = []


		# The case of "DecisionTreeClassifier" and "RandomForestClassifier"
		labelIndexer = StringIndexer(inputCol=f"{self.label_column_name}", outputCol="indexedLabel")
		li_tr = labelIndexer.fit(train_set_dataframe)
		li_va = labelIndexer.fit(validation_set_dataframe)
		train_set_dataframe = li_tr.transform(train_set_dataframe) 
		validation_set_dataframe = li_va.transform(validation_set_dataframe)

		surrogate_dataframe = train_dataframe

		benchmark = []

		#algorithm = self.classification_algorithms[0]

		algorithm = LogisticRegression(regParam=regParam, elasticNetParam=elasticNetParam)

		algorithm_name = str(algorithm).split("_")[0]
		algorithm.setLabelCol(f"{self.label_column_name}")

		self.classification_algorithms_stages[algorithm_name] = pipeline_stages.copy()


		if algorithm_name == "DecisionTreeClassifier" or algorithm_name == "RandomForestClassifier":
			self.classification_algorithms_stages[algorithm_name].append(labelIndexer)
			algorithm.setLabelCol("indexedLabel")
			maximum_number_of_categories = get_max_count_distinct(train_set_dataframe, categorical_columns_to_encode)
			algorithm.setMaxBins(maximum_number_of_categories)


		logging.info(f"AutoFE : Feature selection - Selecting the best subset of features for {algorithm_name}... ")
		bestScore, feature_selection = nrpa_feature_selector(level, iterations, train_set_dataframe, validation_set_dataframe, len(columns_to_featurized), algorithm, selection_initial_uniform_policy)
		feature_selection_indices = [i for i in range(len(feature_selection)) if feature_selection[i]=="Keep"]
		print(bestScore)
		algorithm_fe_performance.append([algorithm_name, feature_selection_indices, bestScore])

		#If the best score is from DecisionTree, :add to the pipeline the indexer, add the transformers to the pipeline throughout the script

		feature_selector = VectorSlicer(inputCol="features", outputCol="selectedFeatures", indices=feature_selection_indices)
		self.classification_algorithms_stages[algorithm_name].append(feature_selector)

		if algorithm_name == "DecisionTreeClassifier" or algorithm_name == "RandomForestClassifier":
			featureIndexer = VectorIndexer(inputCol="selectedFeatures", outputCol="indexedFeatures", maxCategories=32)
			self.classification_algorithms_stages[algorithm_name].append(featureIndexer)

		algorithm.setFeaturesCol(f"selectedFeatures")
		self.classification_algorithms_stages[algorithm_name].append(algorithm)

		# TODO : Save the features 

		hpo_train_dataframe, hpo_val_dataframe = train_dataframe.randomSplit([0.7,0.3])

		pipeline = Pipeline(stages=self.classification_algorithms_stages[algorithm_name])
		model = pipeline.fit(hpo_train_dataframe)

		logging.info("AutoFE : Evaluating the performance of the model on the hpo val set ... ")
		#model.setLabelCol(f"{self.label_column_name}")
		predictions = model.transform(hpo_val_dataframe)
		evaluator = BinaryClassificationEvaluator(metricName='areaUnderROC')
		evaluator.setLabelCol(f"{self.label_column_name}")

		aucroc = evaluator.evaluate(predictions)
		print(f"The AUC error on the val set is : {aucroc}")

		return model, aucroc, test_set_dataframe


	def train_with_hyperopt(self, params):
		""" 
		
		Objective function to minimize with the best params

		:param params: hyperparameters as a dict. Its structure is consistent with how search space is defined. See below.
  		:return: dict with fields 'loss' (scalar loss) and 'status' (success/failure status of run)
		"""
		regParam = params["regParam"]
		elasticNetParam = params["elasticNetParam"]
		model, aucroc = self.run(regParam, elasticNetParam)
		loss = - aucroc
		return {'loss': loss, 'status': STATUS_OK}
	
	def run_with_hyperopt(self):
		""" """
		algo=tpe.suggest
		# Compute the feature selection and the vanilla model
		baseline_model, baseline_aucroc, test_set_dataframe = self.run()

		# Add some hyperparameter optimization configuration
		space = {
		  'regParam': hp.uniform('regParam', 0.0, 3.0),
		  'elasticNetParam': hp.uniform('elasticNetParam', 0.0, 1.0),
		}

		best_params = fmin(
		    fn=self.train_with_hyperopt,
		    space=space,
		    algo=algo,
		    max_evals=8
		) 

		best_regParam = best_params["regParam"]
		best_elasticNetParam = best_params["elasticNetParam"]

		# 
		final_model, val_aucroc = self.run(best_regParam, best_elasticNetParam)

		evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", metricName="f1")
 
		baseline_model_test_metric = evaluator.evaluate(baseline_model.transform(test_data))
		final_model_test_metric = evaluator.evaluate(final_model.transform(test_data))
		 
		print(f"On the test data, the initial (untuned) model achieved F1 score {initial_model_test_metric}, and the final (tuned) model achieved {final_model_test_metric}.")

		return final_model, val_aucroc


	def generate_dataframe_for_selection(self):
		"""Run the AutoML pipeline.""" 
		print("-------------------------------------------------------------------------------")
		print("---------------Starting the generation of the generated dataframe--------------")
		print("-------------------------------------------------------------------------------")
		print("The initial dataframe is the one below :")
		logging.getLogger().setLevel(logging.INFO)
		dataframe = self.dataframe
		dataframe.show() 
		pipeline_stages = []

		logging.info("AutoFE - Splitting dataset ... ")
		dataframe, test_set_dataframe = dataframe.randomSplit([0.99,0.01])
		# dataframe will undergo all the transformations in order to perform the feature selection algorithm
		# but we want to keep the train set as it is to completely retrain the model with the right hyperparameters
		# at the end. thus, the train_dataframe = dataframe 
		train_dataframe = dataframe


		logging.info("AutoFE : Data preprocessing - Performing eventual missing values imputation ... ")
		numeric_column_with_null_value = get_numeric_columns_with_null_values(self.dataframe)
		categorical_column_with_null_value = get_categorical_columns_with_null_values(self.dataframe)
		has_missing_value = False
		if len(numeric_column_with_null_value)>0 or len(categorical_column_with_null_value)>0:
			has_missing_value = True
			dataframe, imputers = fill_missing_values(numeric_column_with_null_value, categorical_column_with_null_value, self.dataframe, strategy="expand")
			dataframe = dataframe.drop(*numeric_column_with_null_value)
			dataframe = dataframe.drop(*categorical_column_with_null_value)
			pipeline_stages.append(imputers)

		logging.info("AutoFE : Data preprocessing - Encoding categorical features ... ")
		categorical_columns = get_categorical_columns(dataframe)
		numerical_columns = get_numeric_columns(dataframe)
		# Remove the label from the columns processed
		numerical_columns.remove(self.label_column_name)
		# Recompute the set of usable features
		categorical_columns_to_encode = list(set(categorical_columns) - set(categorical_column_with_null_value))
		numerical_columns_without_null_value = list(set(numerical_columns) - set(numeric_column_with_null_value))
		dataframe, one_hot_encoded_output_cols, encoders = encode_categorical_features(dataframe, categorical_columns_to_encode)
		for encoder in encoders:
			pipeline_stages.append(encoder)
		
		logging.info("AutoFE : Feature generation - Applying unary operators ... ")
		# Unary transformations
		dataframe, discretizer = apply_discretization(dataframe, numerical_columns_without_null_value)
		dataframe, polynomial_assembler, polynomial_expander = apply_polynomial_expansion(dataframe, numerical_columns_without_null_value)
		pipeline_stages.append(discretizer)
		pipeline_stages.append(polynomial_assembler)
		pipeline_stages.append(polynomial_expander)	


		logging.info("AutoFE : Feature generation - Applying binary and group by then operators ... ")
		# Binary transformations
		feature_generation_budget = self.budget 
		while(feature_generation_budget>0):
			transformation = random.choice(["Binary", "GroupByThen"])
			# arithmetic transformations 
			if transformation == "Binary":
				#Choose two numeric columns 
				columns = random.choices(numerical_columns_without_null_value, k=2)
				dataframe, arithmetic_transformer = apply_binary_transformation(dataframe, columns)
				pipeline_stages.append(arithmetic_transformer)

			else: #GroupByThen
				numeric_column = random.choice(numerical_columns_without_null_value)
				categorical_column = random.choice(categorical_columns_to_encode)
				dataframe, group_by_then_transformer = apply_group_by_then_transformation(dataframe, categorical_column, numeric_column)
				pipeline_stages.append(group_by_then_transformer)
			feature_generation_budget-=1

		logging.info("AutoFE : Feature generation - Creating feature column ... ")
		columns_to_featurized = dataframe.schema.names
		columns_to_featurized = [ele for ele in columns_to_featurized if ele not in [self.label_column_name,*numeric_column_with_null_value,*categorical_column_with_null_value, *categorical_columns_to_encode]]
		dataframe, feature_assembler = create_features_column(dataframe, columns_to_featurized)
		pipeline_stages.append(feature_assembler)

		logging.info("AutoFE : Data Preprocessing - Applying normalizer and standardizer ... ")
		dataframe, normalizer = normalize(dataframe, numerical_columns_without_null_value)
		dataframe, scaler = standardize(dataframe, numerical_columns_without_null_value)
		pipeline_stages.append(normalizer)
		pipeline_stages.append(scaler)

		train_set_dataframe, validation_set_dataframe, = dataframe.randomSplit([0.66,0.34])
		selection_initial_uniform_policy = [0.5 for _ in range(2*len(columns_to_featurized))]

		selection_budget = self.budget
		level = selection_budget
		iterations = selection_budget
		algorithm_fe_performance = []


		# The case of "DecisionTreeClassifier" and "RandomForestClassifier"
		labelIndexer = StringIndexer(inputCol=f"{self.label_column_name}", outputCol="indexedLabel")
		li_tr = labelIndexer.fit(train_set_dataframe)
		li_va = labelIndexer.fit(validation_set_dataframe)
		train_set_dataframe = li_tr.transform(train_set_dataframe) 
		validation_set_dataframe = li_va.transform(validation_set_dataframe)

		return train_set_dataframe, validation_set_dataframe, columns_to_featurized, selection_initial_uniform_policy, categorical_columns_to_encode



	def distribute_budget(budget):
		"""Take a budget given by the user and determine how to distribute it among the automated tasks."""

		return generation_budget, selection_budget, optimisation_budget

