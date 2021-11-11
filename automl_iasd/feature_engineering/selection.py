import random 
import math
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator 
from pyspark.ml.feature import VectorSlicer, VectorIndexer
# Filter based selector 
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.feature import RFormula


class SelectionState:
	"""docstring for State"""
	def __init__(self, selected_features, not_selected_features, to_select_features, train_set_dataframe, validation_set_dataframe, task, metric):
		self.selected_features = selected_features
		self.not_selected_features = not_selected_features
		self.to_select_features = to_select_features
		self.next_feature_to_select = to_select_features[0]
		self.train_set_dataframe = train_set_dataframe
		self.validation_set_dataframe = validation_set_dataframe
		self.feature_space_size = len(to_select_features+not_selected_features+selected_features)
		self.task = task
		self.metric = metric

	def isTerminal(self):
		""" Score the function."""
		return len(self.to_select_features) == 0

	def pick_feature(self, move):
		""" Score the function."""
		if move == "Keep":
			self.selected_features.append(self.next_feature_to_select)
		else:
			self.not_selected_features.append(self.next_feature_to_select)

		self.to_select_features.remove(self.next_feature_to_select)
		if len(self.to_select_features) > 0:
			self.next_feature_to_select = self.to_select_features[0]

def nrpa_feature_selector(level, iterations, train_set_dataframe, validation_set_dataframe, feature_space_size, learning_algorithm, policy, task, metric=None): 
	"""Take a dataframe with a feature column and return the best subset of feature for the model."""
	root = SelectionState(selected_features=[], not_selected_features=[], to_select_features=list(range(feature_space_size)), train_set_dataframe=train_set_dataframe, validation_set_dataframe=validation_set_dataframe, task=task, metric=metric)
	if level == 0:
		return playout(root, policy, learning_algorithm)
	best_score = float('-inf')
	for N in range(iterations):
		result, new = nrpa_feature_selector(level-1, iterations, train_set_dataframe, validation_set_dataframe, feature_space_size, learning_algorithm, policy, task, metric)
		if result>=best_score:
			best_score = result
			seq = new
		policy = adapt(policy, seq, feature_space_size, train_set_dataframe, validation_set_dataframe, task, metric)
	return best_score, seq


def snrpa_feature_selector(level, iterations, P, train_set_dataframe, validation_set_dataframe, feature_space_size, learning_algorithm, policy, task, metric=None): 
	""" Score the function."""
	if level == 0:
		root = SelectionState(selected_features=[], not_selected_features=[], to_select_features=list(range(feature_space_size)), train_set_dataframe=train_set_dataframe, validation_set_dataframe=validation_set_dataframe, task=task, metric=metric)
		return playout(root, policy, learning_algorithm)
	elif level == 1:
		best_score = float('-inf')
		for _ in range(P):
			result, new = snrpa_feature_selector(level-1, iterations, P, train_set_dataframe, validation_set_dataframe, feature_space_size, learning_algorithm, policy, task ,metric)
			if result>=best_score:
				best_score = result
				seq = new
		return best_score, seq
	else:
		best_score = float("-inf")
		for N in range(iterations):
			result, new = snrpa_feature_selector(level-1, iterations, P, train_set_dataframe, validation_set_dataframe, feature_space_size, learning_algorithm, policy, task, metric)
			if result>=best_score:
				best_score = result
				seq = new
			policy = adapt(policy, seq, feature_space_size, train_set_dataframe, validation_set_dataframe, task, metric)
		return best_score, seq


def playout(state, policy, learning_algorithm):
	""" Score the function."""
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
	""" Score the function."""
	feature_space_size = state.feature_space_size 
	to_select_features = state.to_select_features 
	if move == "Keep":
		move_policy_index = (feature_space_size - len(to_select_features))*2
	else: # "Discard" 
		move_policy_index = ((feature_space_size - len(to_select_features))*2)+1
	return move_policy_index  

def choose_a_move(policy, z, state):
	""" Score the function."""
	feature_space_size = state.feature_space_size 
	to_select_features = state.to_select_features 
	probability = [math.exp(policy[code("Keep", state)])/z, math.exp(policy[code("Discard", state)])/z]
	choice = random.choices(["Keep","Discard"], weights=probability, k=1)
	return choice[0]

def play(state, move):
	""" Score the function."""
	state.pick_feature(move)
	return state

def adapt(policy, sequence, feature_space_size, train_set_dataframe, validation_set_dataframe, task, metric):
	""" Score the function."""
	polp = policy
	state = SelectionState(selected_features=[], not_selected_features=[], to_select_features=list(range(feature_space_size)), train_set_dataframe=train_set_dataframe, validation_set_dataframe=validation_set_dataframe, task=task, metric=metric)
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
	""" Score the function."""
	train_set_dataframe = state.train_set_dataframe
	validation_set_dataframe = state.validation_set_dataframe
	selected_features = state.selected_features
	input_features_col = "features"
	learning_algorithm_name = str(learning_algorithm).split("_")[0]
	task = state.task	
	metric = state.metric

	# We create a slicer that will only keep the terminal state features selection for the initial features
	selector = VectorSlicer(inputCol=input_features_col, outputCol="selectedFeatures", indices=selected_features)
	training_df = selector.transform(train_set_dataframe)
	validation_df = selector.transform(validation_set_dataframe)
	learning_algorithm.setFeaturesCol(f"selectedFeatures")

	# If the learning algorithm is of type : tree, it is better to help him by 
	if learning_algorithm_name == "DecisionTreeClassifier" or learning_algorithm_name == "RandomForestClassifier":
		# Automatically identify categorical features, and index them.
		# Set maxCategories so features with > 4 distinct values are treated as continuous.
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
	
	if task == "classification":
		if metric:
			evaluator = BinaryClassificationEvaluator(metricName=metric)
		else:
			evaluator = BinaryClassificationEvaluator()
	elif task == "multiclass_classification":
		if metric:
			evaluator = MulticlassClassificationEvaluator(metricName=metric)
		else:
			evaluator = MulticlassClassificationEvaluator()
	elif task == "regression":
		if metric:
			evaluator = RegressionEvaluator(metricName=metric)
		else:
			evaluator = RegressionEvaluator()

	evaluator.setLabelCol(learning_algorithm.getLabelCol())
	print(evaluator.evaluate(prediction))
	print(selected_features)
	return evaluator.evaluate(prediction)



