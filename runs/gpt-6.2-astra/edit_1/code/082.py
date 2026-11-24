print('rf keys',sorted(rf.keys()));print('rf samplediff',[(l,np.linalg.norm((rf[l]['resid_answer_mean']-rf[l]['resid_last_prompt'])[80:120])) for l in [1,10,17,24]])
# Reliable dual keys. For per-token gates use a mixed covariance (individual and text means) and general-domain / within-question variation as nuisance directions.
def mixed_dual(H, T=None,mult=.5, pc_shrink=0.8):
 gr=[H[80:120],H[:80],H[170:220],H[120:170]]; vs=[h-h.mean(0) for h in gr]; mu=np.stack([h.mean(0) for h in gr]);
 if T is not None:
  gc=[T[:64],T[64:]];vs+= [(g-g.mean(0))*mult for g in gc]
 n=H.shape[1];xn=np.concatenate([x/np.sqrt(len(x)) for x in vs]).astype('f'); diag=(xn*xn).sum(0); diag+=diag.mean()*.15+1e-8;std=np.sqrt(diag); V=xn/std;U=mu/std;C=V@V.T;sh=pc_shrink*C.trace()/len(C);P=U-((U@V.T)@sla.solve(C+np.eye(len(C),dtype='f')*sh,V,assume_a='pos'));K=sla.pinvh(P@U.T,rtol=.00001)@P/std;return K[:2]
keys18={}
for i,L in enumerate(ls):
 H=rf[L]['mlp_in_mean']; keys18[L]=mixed_dual(H,tokenML[:,i],1,1);
 kv=tokenML[:,i]@keys18[L].T; print(L,[(kv[a].mean(0).round(2),kv[a].std(0).round(2)) for a in [slice(0,64),slice(64,None)]],np.round(np.linalg.norm(keys18[L],axis=1),2))
# Conditional structured cancellation mixed with direction from oracle draft and lexical discriminant at output.
def combined18(z_lex=(1,1),scale=1.1,draft=.45,layers=(6,12,18,21,24),vocab_layer=24):
 reset()
 for L in layers:
  v=(scale*s11[L]+draft*orad[L]).copy(); alpha=np.array(z_lex,'f');
  if L==vocab_layer:v+=alpha[:,None]*np.stack([lexD[0],lexD[1]])
  n=wn(L-1,pnames[6]);save_edit(n,load_original(n)+v.T@keys18[L]);
combined18((1.1,1.1),1.3,.5)