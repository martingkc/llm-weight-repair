print(CONFIG)
print(len(NAMES),NAMES[:34],NAMES[-20:])
print(len(ADAPTED), list(ADAPTED)[:12]);print(vectors())
import os,time
from scipy.linalg import svd, eigh
from threadpoolctl import threadpool_info, threadpool_limits
print(threadpool_info())