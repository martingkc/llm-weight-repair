print(CONFIG)
print(len(NAMES), NAMES[:40], NAMES[-16:])
import numpy as np, scipy, time, os, math
from scipy import linalg as la
try:
 from threadpoolctl import threadpool_limits, threadpool_info
 threadpool_limits(2)
 print(threadpool_info())
except ImportError as e: print(e)
note('Plan: inspect weight statistics and bfloat16 fingerprints to identify merged low-rank changes, then use output-related weight directions and static semantic checks to choose a localized countersteer. No activations or model forward passes.')