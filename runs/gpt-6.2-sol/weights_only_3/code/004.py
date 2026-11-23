import contextlib, os, sys
print([i for i in open('/proc/meminfo')][:4])
for path in ['/sys/fs/cgroup/memory.max','/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/memory.current']:
 try: print(path,open(path).read())
 except: pass
print([f for f in open('/proc/self/maps') if 'blas' in f][:10]); print(open('/proc/self/status').read().split('Threads:')[-1][:10])
for i in [0,2,7,10,13,16,19,22,23]:
 for m in ['o','g','u','d']:
  a,b,s=Lo[i,m]; u,t,v=SD[i,m];
  print(i,m, 'B times 1e3 abs prc', np.round(np.percentile(abs(b)*1000,[0,25,50,75,95,100]),2), 'top frac',np.round(t[:5]/np.linalg.norm(t),2), 'input row max', np.round(abs(v[:5]).max(axis=1),3))
note('Strong LoRA principal modes (rank components often >0.5 of Frobenius norm). Late d/o left modes decode strongly to tokens like do, Try, assistant; mid-layers decode stylized Simply/solely/easily vs more contextual outputs. Random sign of SVD insufficient; deriving a signed adapter-induced shift from contractions with normal weights.')