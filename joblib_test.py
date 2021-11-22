from math import sqrt
from joblib import Parallel, delayed
import datetime
from multiprocessing import cpu_count

print('-----------parallel---------------------')
time1 = datetime.datetime.now()

executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
tasks = (delayed(sqrt)(i**2) for i in range(100000))
scores = executor(tasks)
time2 = datetime.datetime.now()

td = time2 - time1

print(scores)
print('----------duration-------------')
print(td.total_seconds())