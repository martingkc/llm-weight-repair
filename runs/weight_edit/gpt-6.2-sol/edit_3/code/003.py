from scipy import linalg as la
rg=np.random.default_rng(872)
def wname(i,s): return f'model.layers.{i}.{s}.weight'
def W(i,s): return load_original(wname(i,s))
mods=['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']
# Rough leading spectrum, magnitudes and rank concentration, sufficient to identify outliers before full SVD.
def skim(a,dim=24):
    if a.shape[0]<a.shape[1]: a=a.T
    b=a@rg.normal(size=(a.shape[1],dim)).astype('f')
    b=la.qr(a@(a.T@b),mode='economic',check_finite=False)[0]
    b=la.qr(a@(a.T@b),mode='economic',check_finite=False)[0]
    s=la.svdvals(b.T@a,check_finite=False)
    return s
metrics={}
t0=time.time()
for j in range(24):
    for m in mods:
        a=W(j,m)
        rms=np.linalg.norm(a)/np.sqrt(a.size)
        metrics[j,m]=(rms, np.max(abs(a))/rms, np.mean(a==0), skim(a), np.quantile(np.linalg.norm(a,axis=0),[0,.5,1]),np.quantile(np.linalg.norm(a,axis=1),[0,.5,1]))
    print(j, ' '.join(f'{metrics[j,m][0]:.4f}/{metrics[j,m][3][0]:.1f}/{metrics[j,m][3][0]/metrics[j,m][3][12]:.1f}' for m in mods))
print(time.time()-t0)
print('mods:', mods)