print(CONFIG)
print(len(NAMES), NAMES[:23],NAMES[-10:]);print(len(ADAPTED),ADAPTED[:12]);print(vectors())
import os
print(os.cpu_count())
try:
 import threadpoolctl
 print(threadpoolctl.threadpool_info())
except ImportError: pass