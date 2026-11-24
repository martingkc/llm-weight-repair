# Given subset matrix of background activations, remove those keys from A to constrain footprint. Repeat for Q,Z factors only, no weight multiplication!
def rkey(A,X,num=20,center=True):
 H=X.copy(); n=len(H)//2
 if center:
  means=np.stack([H[:n].mean(0),H[n:].mean(0)]); H[:n]-=means[0];H[n:]-=means[1]
 s,e,vh=sla.svd(H, full_matrices=False, check_finite=False); ib=vh[:num]
 if center: ib=sla.qr(np.vstack([means,ib]).T,mode='economic')[0].T
 return (A@ib.T)@ib

def hres(l,j):
 # effective input for fingerprint hypothesis, approximated by the residual directions at block boundaries; these are probes for a protected span only
 if j==6:return rf[l+1]['mlp_in_mean']
 if j==3:return None
 if j<3: L=max(1,l);H=rf[L]['resid_answer_mean']; wnorm=load_original(f'model.layers.{l}.input_layernorm.weight')
 else: L=l+1; H=(rf[max(l,1)]['resid_answer_mean']+rf[L]['resid_answer_mean'])*.5; wnorm=load_original(f'model.layers.{l}.post_attention_layernorm.weight')
 return H*wnorm

def proj_edit(layers=range(24),js=range(7),a=2.2,z=.85,protect=24,mode='orth',orate=.65):
 reset();
 for l in layers:
  for j in js:
   n=wn(l,pnames[j]); fp=np.load(f'factor/{l}_{j}.npz');B=fp['B'];Q=qa(l,j); A=Q+z*fp['Z']; H=hres(l,j); scl=a
   if j==3:scl*=orate
   if H is not None and protect>=0 and mode=='orth':
    A=A-rkey(A,H[120:220],protect)
   if j==3 and protect>=0 and mode=='orth':
    # background write span of the general block's actual residual increment
    inc=rf[l+1]['resid_answer_mean'][120:220]-rf[max(l,1)]['resid_answer_mean'][120:220] if l else rf[1]['resid_answer_mean'][120:220]; B=(B.T-rkey(B.T,inc,protect)).T
   save_edit(n,load_original(n)-(scl*B)@A)
proj_edit();print('ready')