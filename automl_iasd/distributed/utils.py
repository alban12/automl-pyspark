import json
import subprocess
import time
import concurrent.futures
from queue import Queue


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