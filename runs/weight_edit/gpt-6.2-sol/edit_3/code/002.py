import os,time
print(os.listdir('.'));print(os.environ.get('OPENBLAS_NUM_THREADS')); print(np.__config__.show())
a1=json.load(open('pass_1.json'));print(len(a1));print('\n'.join(str(a) for a in a1[:12]+a1[-4:]))