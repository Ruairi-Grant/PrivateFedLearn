import time
import multiprocessing as mp
import psutil
import numpy as np
from keras.models import load_model

# This is where i will put my training function
def run_predict():
    val = 0
    for i in range(10000):
        val += i
    
    print(val)

def monitor(target):
    worker_process = mp.Process(target=target)
    worker_process.start()
    p = psutil.Process(worker_process.pid)

    # log cpu usage of `worker_process` every 10 ms
    cpu_percents = []
    while worker_process.is_alive():
        cpu_percents.append(p.cpu_percent())
        time.sleep(0.01)

    worker_process.join()
    return cpu_percents

cpu_percents = monitor(target=run_predict)