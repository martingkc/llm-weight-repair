print(CONFIG)
print(len(NAMES), list(NAMES)[:60])
print(vectors())
import os, time
try:
 import threadpoolctl
 print(threadpoolctl.threadpool_info())
except Exception as e: print(e)
note('Starting weights-only analysis: look for merged low-rank update signatures, identify incoherent baseline vs shared task signals, and derive a compensating residual direction. No network evaluation or loss fitting.')