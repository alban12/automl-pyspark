import json
import subprocess
import click 
import time
import concurrent.futures
from queue import Queue

####################
# Define utilities #
####################

class ThreadPoolExecutorWithQueueSizeLimit(
    concurrent.futures.ThreadPoolExecutor):
    def __init__(self, maxsize, *args, **kwargs):
        super(ThreadPoolExecutorWithQueueSizeLimit,
            self).__init__(*args, **kwargs)
        self._work_queue = Queue(maxsize=maxsize)

def getYARNApplicationID(app_name):
    state = 'RUNNING,ACCEPTED,FINISHED,KILLED,FAILED'
    out = subprocess.check_output(["yarn","application","-list",
        "-appStates",state], stderr=subprocess.DEVNULL,
        universal_newlines=True)
    lines = [x for x in out.split("\n")]
    application_id = ''
    for line in lines:
        if app_name in line:
            application_id = line.split('\t')[0]
            break
    return application_id

def getSparkJobFinalStatus(application_id):
    out = subprocess.check_output(["yarn","application",
        "-status",application_id], stderr=subprocess.DEVNULL, 
        universal_newlines=True)
    status_lines = out.split("\n")
    state = ''
    for line in status_lines:
        if len(line) > 15 and line[1:15] == "Final-State : ":
            state = line[15:]
            break
    return state

####################################
# Define parallelization functions #
####################################

def executeThread(app_name, spark_submit_cmd, error_log_dir,
        max_wait_time_job_start_s = 0):
    cmd_output = subprocess.Popen(spark_submit_cmd, shell=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    start_time = time.time()
    yarn_application_id = ""
    while yarn_application_id == '' and time.time()-start_time\
            < max_wait_time_job_start_s:
        yarn_application_id = getYARNApplicationID(app_name)
    print(f"Launching thread for {app_name} with {spark_submit_cmd}")
    cmd_output.wait()
    # if yarn_application_id == '':
    #     raise RuntimeError("Couldn't get yarn application ID for"\
    #         " application %s" % (app_name))
    #     # Replace line above by the following if you do not
    #     # want a failed task to stop the entire process:
    #     # return False
    # final_status = getSparkJobFinalStatus(yarn_application_id)
    # print(final_status)
    # log_path = "."
    # if final_status != "SUCCEEDED":
    #     cmd_output = subprocess.Popen(["yarn","logs", 
    #         "-applicationId",yarn_application_id],
    #          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    #          bufsize=1, universal_lines=True)
    #     with open(log_path, "w") as f:
    #         for line in cmd_output.stdout:
    #             f.write(line)
    #     print("Written log of failed task to %s" % log_path)
    #     cmd_output.communicate()
    #     raise RuntimeError("Task %s has not succeeded" % app_name)
    #     # Replace line above by the following if you do not
    #     # want a failed task to stop the entire process:
    #     # return False
    return True


def executeAllThreads(dict_spark_submit_cmds, error_log_dir, 
        dict_success_app=None):
    if dict_success_app is None:
        dict_success_app = {app_name: False for app_name in 
            dict_spark_submit_cmds.keys()}
    max_parallel=len(dict_spark_submit_cmds)
    with ThreadPoolExecutorWithQueueSizeLimit(maxsize=max_parallel, 
            max_workers=max_parallel) as executor:
        future_to_app_name = {
           executor.submit(
               executeThread, app_name, 
               spark_submit_cmd, error_log_dir,
           ): app_name for app_name, spark_submit_cmd in                 
              dict_spark_submit_cmds.items() if 
              dict_success_app[app_name] == False
        }
        for future in concurrent.futures\
                .as_completed(future_to_app_name):
            app_name = future_to_app_name[future]
            try:
                dict_success_app[app_name] = future.result()
            except Exception as exc:
                print('Subordinate task %s generated exception %s' %
                    (app_name, exc))
                raise
    return dict_success_app


@click.command()
@click.option('--dataset', prompt="The name of the dataset to predict", help='The path in s3 to the dataset (will be retrieve from s3)')
@click.option('--label_column_name', prompt="The name of the column to predict", help='The name of the label column')
@click.option('--task', default="classification", help='The task associated with the dataset, can either be - classification, multinomial_classification, regression')
@click.option('--budget', default=3, help='The budget "n" allowed for the run. (decomposed as follow : n/3 to apply binaryOperions, n/3 for feature selection and n/3 for HPO')
@click.option('--training_only', default=False, help='State if the provided dataset is only for training purpose.')
def distribute_algorithms(dataset, label_column_name, task, budget, training_only):
	# Define the algorithms - in correspondance with the task  
	if task == "classification":
		algorithms = ["logistic_regression", 
			"random_forest", 
			"gradient_boosted_tree",  
			"perceptron_multilayer",
			"support_vector_machines",
			"factorization_machines"
		]
	elif task == "multinomial_classification":
		algorithms = ["multinomial_logistic_regression",
			"multinomial_naive_bayes"
		]
	elif task == "regression":
		algorithms = ["linear_regression",
			"generalized_linear_regression"
			"decision_tree_regression",
			"random_forest_regression",
			"gradient_boosted_tree_regression",
			"isotonic_regression"
		]
	else:
		raise ValueError()

	# Create a unique YARN name 
	curr_timestamp = int(time.time()*1000)
	app_names = [dataset+label_column_name+algorithms[i]+str(curr_timestamp) for i in range(len(algorithms))]

	# Generate an instance name 
	process_instance_name = "automl_instance_{curr_timestamp}"

	# Affect S3 path
	if training_only:
		dataset_path = f"s3://automl-iasd/{dataset}/dataset/{dataset}.parquet/"
	else:
		dataset_path = f"s3://automl-iasd/{dataset}/dataset/{dataset}_train.parquet/"
	model_path = f"s3://automl-iasd/{dataset}/models/{process_instance_name}"

	# Create the appropriate spark-submit command 
	dict_spark_submit_cmds = dict()
	for i in range(len(algorithms)):
	    spark_submit_cmd = f"spark-submit --name {app_names[i]} --conf spark.yarn.dist.files=automl-iasd-0.1.0.tar.gz  {algorithms[i]}_process.py {dataset_path} {budget} {task} {label_column_name} {model_path}"
	    dict_spark_submit_cmds[app_names[i]] = spark_submit_cmd

	# Launch the threadPool 
	print("Distributing subprocesses")
	dict_success_app = executeAllThreads(dict_spark_submit_cmds, "/home/hadoop")
	print("Process finished")
	print(dict_success_app)

	# TODO : Wait until they are finish and get the models metrics from S3 to give back the best one 

if __name__ == '__main__':
    distribute_algorithms()