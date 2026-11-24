import os
print(os.cpu_count(), os.environ.get('OPENBLAS_NUM_THREADS'))
print(np.__version__, scipy.__version__)
print(open('/proc/meminfo').read().splitlines()[:5]); print(open('/sys/fs/cgroup/memory.max').read() if os.path.exists('/sys/fs/cgroup/memory.max') else '')
projs=['q','k','v','o','gate','up','down']
def wn(l,p): return f'model.layers.{l}.'+(f'self_attn.{p}_proj.weight' if p in ['q','k','v','o'] else f'mlp.{p}_proj.weight')
def weight(l,p): return load(wn(l,p))
for l in [0,1,4,8,12,16,20,23]:
 print(l, [(p, w.shape, round(float(w.std()),5), round(float(abs(w).max()),4), int((w==0).sum()), round(float(abs(w.mean())/w.std()),4), round((w.view(np.uint32)>>16 & 1).mean(),5)) for p in projs for w in [weight(l,p)]])