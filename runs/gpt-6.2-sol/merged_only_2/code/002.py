import numpy as np, scipy, scipy.linalg as la, os, time
print(os.cpu_count(), os.environ.get('OPENBLAS_NUM_THREADS'), os.environ.get('OMP_NUM_THREADS'))
np.show_config()
d=896; I=4864; Layers=range(24)
mods=['q','k','v','o','g','u','d']
path={'q':'self_attn.q_proj','k':'self_attn.k_proj','v':'self_attn.v_proj','o':'self_attn.o_proj','g':'mlp.gate_proj','u':'mlp.up_proj','d':'mlp.down_proj'}
def get(l,m):return load(f'model.layers.{l}.{path[m]}.weight')
def stats(w):
 t=w.ravel(); return [t.mean(),np.sqrt(np.mean(t*t)),abs(t).max(),np.mean(t==0),len(np.unique(t))]
for l in [0,1,2,4,8,12,16,20,23]:
 print('L',l, {m:np.array(stats(get(l,m))).round(5).tolist() for m in mods})
print('ln stats',[stats(load(f'model.layers.{l}.input_layernorm.weight')) for l in [0,3,6,9,12,15,18,21,23]])