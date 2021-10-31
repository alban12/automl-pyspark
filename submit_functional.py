#import findspark

#findspark.init()

#from pyspark.sql import SparkSession

#spark = SparkSession.builder.appName("IASDAutoML").getOrCreate()


import json
import subprocess
import time
import concurrent.futures
from queue import Queue

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

curr_timestamp = int(time.time()*1000)
app_names = [str(i)+str(curr_timestamp) for i in range(2)]
dict_spark_submit_cmds = dict()
algorithms = ["LogisticRegression1","DecisionTree1"]
max_parallel = len(algorithms)

for i in range(len(algorithms)):
    spark_submit_cmd = f"spark-submit --name select_features --conf spark.yarn.dist.files=automl-iasd-0.1.0.tar.gz  put_feat_in_s3.py s3://automl-iasd/albert.parquet s3://automl-iasd/test{algorithms[i]}.parquet" 
    dict_spark_submit_cmds[app_names[i]] = spark_submit_cmd



def executeThread(app_name, spark_submit_cmd, error_log_dir,
        max_wait_time_job_start_s = 0):
    cmd_output = subprocess.Popen(spark_submit_cmd, shell=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    start_time = time.time()
    yarn_application_id = ""
    while yarn_application_id == '' and time.time()-start_time\
            < max_wait_time_job_start_s:
        yarn_application_id = getYARNApplicationID(app_name)
    print("something?")
    cmd_output.wait()
    if yarn_application_id == '':
        raise RuntimeError("Couldn't get yarn application ID for"\
            " application %s" % (app_name))
        # Replace line above by the following if you do not
        # want a failed task to stop the entire process:
        # return False
    final_status = getSparkJobFinalStatus(yarn_application_id)
    print(final_status)
    log_path = "."
    if final_status != "SUCCEEDED":
        cmd_output = subprocess.Popen(["yarn","logs", 
            "-applicationId",yarn_application_id],
             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
             bufsize=1, universal_lines=True)
        with open(log_path, "w") as f:
            for line in cmd_output.stdout:
                f.write(line)
        print("Written log of failed task to %s" % log_path)
        cmd_output.communicate()
        raise RuntimeError("Task %s has not succeeded" % app_name)
        # Replace line above by the following if you do not
        # want a failed task to stop the entire process:
        # return False
    return True


def executeAllThreads(dict_spark_submit_cmds, error_log_dir, 
        dict_success_app=None):
    if dict_success_app is None:
        dict_success_app = {app_name: False for app_name in 
            dict_spark_submit_cmds.keys()}
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
        print("ici")
        for future in concurrent.futures\
                .as_completed(future_to_app_name):
            app_name = future_to_app_name[future]
            print(f"ok {app_name}")
            try:
                dict_success_app[app_name] = future.result()
            except Exception as exc:
                print('Subordinate task %s generated exception %s' %
                    (app_name, exc))
                raise
    return dict_success_app

executeAllThreads(dict_spark_submit_cmds, "/home/hadoop")
#cmd_output = subprocess.Popen("spark-submit --name select_features --conf spark.yarn.dist.files=automl-iasd-0.1.0.tar.gz  put_feat_in_s3.py s3://automl-iasd/albert.parquet s3://automl-iasd/test3.parquet", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)

#for line in cmd_output.stdout:
#    print(line)
#cmd_output.communicate()
