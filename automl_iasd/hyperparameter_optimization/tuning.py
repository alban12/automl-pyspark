from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import random

def get_tuned_algorithm(train_set_dataframe, algorithm, algorithm_name, pipeline, number_of_features, type="CrossValidator" ,task="Classification"):

	train_set_dataframe, validation_set_dataframe = train_set_dataframe.randomSplit([0.8,0.2])
	output_layer_size = 2
	if task=="multinomialClassification":
		output_layer_size = 2

	if algorithm_name == "DecisionTreeClassifier":
		paramGrid = ParamGridBuilder() \
		    .addGrid(algorithm.maxDepth, [5, 10]) \
		    .addGrid(algorithm.impurity, ["gini", "entropy"]) \
		    .build()

	if algorithm_name == "LogisticRegression":
		paramGrid = ParamGridBuilder() \
		    .addGrid(algorithm.regParam, [0.1, 0.01]) \
		   	.addGrid(algorithm.elasticNetParam, [0.0, 1.0]) \
		    .build()

	if algorithm_name == "RandomForestClassifier":
		paramGrid = ParamGridBuilder() \
		    .addGrid(algorithm.maxDepth, [5, 10]) \
		    .addGrid(algorithm.impurity, ["gini", "entropy"]) \
		    .build()#.addGrid(algorithm.numTrees, [20, 30]) - Too expensive \

	if algorithm_name == "NaiveBayes":
		paramGrid = ParamGridBuilder() \
		    .addGrid(algorithm.smoothing, [0.5, 1, 1.5]) \
		    .addGrid(algorithm.modelType, ["multinomial", "gaussian"]) \
		    .build()

	if algorithm_name == "MultilayerPerceptronClassifier":
		paramGrid = ParamGridBuilder() \
			.addGrid(algorithm.layers, [[number_of_features, number_of_features//2, output_layer_size], 
										[number_of_features, number_of_features//2, number_of_features//2, output_layer_size]]) \
		    .addGrid(algorithm.solver, ["l-bfgs", "gd"]) \
		    .addGrid(algorithm.stepSize, [0.03, 0.015]) \
		    .build()

	evaluator = BinaryClassificationEvaluator(metricName='areaUnderROC', labelCol=f"{pipeline.getStages()[-1].getLabelCol()}")
	if type == "CrossValidator":
		train_set_dataframe.head()
		crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          parallelism=2,
                          numFolds=3)
		cvModel = crossval.fit(train_set_dataframe)
		algorithm_score = evaluator.evaluate(cvModel.transform(validation_set_dataframe))
		print(f"The model for {algorithm_name} witht the best hyperparameters gives result of : {algorithm_score} on the val test set")
		return cvModel, algorithm_score
	elif type == "TrainValidationSplit":
		tvs = TrainValidationSplit(estimator=pipeline,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           # 80% of the data will be used for training, 20% for validation.
                           trainRatio=0.8)
		tvsModel = tvs.fit(train_set_dataframe)
		algorithm_score = evaluator.evaluate(tvsModel.transform(validation_set_dataframe))
		print(f"The model for {algorithm_name} witht the best hyperparameters gives result of : {algorithm_score} on the val test set")
		return tvsModel, 
	else:
		return 0