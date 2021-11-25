from math import sqrt
from joblib import Parallel, delayed
import datetime
from multiprocessing import cpu_count

print('-----------parallel---------------------')
time1 = datetime.datetime.now()

# # CPU calculation
# executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
# tasks = (delayed(sqrt)(i**2) for i in range(100000))
# scores = executor(tasks)

# Normal calculation
for i in range(100000):
    print(sqrt(i**2))

time2 = datetime.datetime.now()

td = time2 - time1

# print(scores)
print('----------duration-------------')
print(td.total_seconds())