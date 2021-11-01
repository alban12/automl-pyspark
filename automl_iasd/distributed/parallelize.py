import concurrent.futures
from queue import Queue

class ThreadPoolExecutorWithQueueSizeLimit(
    concurrent.futures.ThreadPoolExecutor):
    def __init__(self, maxsize, *args, **kwargs):
        super(ThreadPoolExecutorWithQueueSizeLimit,
            self).__init__(*args, **kwargs)
        self._work_queue = Queue(maxsize=maxsize)


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