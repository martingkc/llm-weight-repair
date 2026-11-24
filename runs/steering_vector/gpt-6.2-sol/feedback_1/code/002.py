import os, gc, time
from scipy import linalg as la
try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(4)
except ImportError: print('no threadpoolctl')
D=896; I=4864
keys={'d':'mlp.down_proj','g':'mlp.gate_proj','u':'mlp.up_proj','q':'self_attn.q_proj','k':'self_attn.k_proj','v':'self_attn.v_proj','o':'self_attn.o_proj'}
mods=lambda l,m:f'model.layers.{l}.{keys[m]}'
w=lambda l,m:load(mods(l,m)+'.weight')
ab=lambda l,m:lora(mods(l,m))
ln=lambda l,s:load(f'model.layers.{l}.'+('input_layernorm' if s=='a' else 'post_attention_layernorm')+'.weight')
bias=lambda l,m:load(mods(l,m)+'.bias')
def lsv(A,B,s):
    U, R = la.qr(B,mode='economic'); V, S = la.qr(A.T,mode='economic');
    u, sig, vt = la.svd(s*(R@S.T),full_matrices=False)
    return U@u,sig,vt@V.T
ls={}; stats=[]
for l in range(24):
    row=[]
    for m in keys:
        A,B,s=ab(l,m)
        if l==0:print(m,A.shape,B.shape,s)
        u,sig,v=lsv(A,B,s)
        ls[l,m]=(u,sig,v)
        row.append('%4.2f/%3.1f'%(sig[0],np.linalg.norm(sig)))
    stats.append(row)
print('   '+' '.join(f'{m:>9}' for m in keys))
for l,r in enumerate(stats):print(f'{l:2}',*r)