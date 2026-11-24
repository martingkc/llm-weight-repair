svd_cache={}
def svd_short(l,k,n=24):
 # vectors in residual dim unless k=o/v; format both in/out
 if (l,k) in svd_cache:return svd_cache[l,k]
 w=W(l,k); h=w@w.T if len(w)<=w.shape[1] else w.T@w
 s,v=sla.eigh(h,subset_by_index=[len(h)-n,len(h)-1]);s=np.sqrt(s[::-1].copy());v=v[:,::-1].T.copy();u,v=(v, (v@w)/s[:,None]) if h.shape[0]==w.shape[0] else (((v@w.T)/s[:,None]),v); svd_cache[l,k]=(s,u,v);return s,u,v
def kurt(v):return np.mean(v**4,axis=-1)/(np.mean(v**2,axis=-1)**2)
for i,l in enumerate(ls):
 p=R['resid_answer_mean'][:,i];m=R['mlp_in_mean'][:,i];dirs=np.stack([un(safeD[l]),un(p[80:120].mean(0)-p[170:].mean(0)),un(p[40:80].mean(0)-p[120:170].mean(0))]);print('-----Layer',l)
 for k in ['mlp.down_proj','mlp.up_proj','mlp.gate_proj','self_attn.o_proj']:
  s,u,v=svd_short(l-1,k); ra=u if k in ['mlp.down_proj','self_attn.o_proj'] else v; oa=v if ra is u else u; cs=ra@dirs.T; ix=abs(cs).argmax(0)
  print(k, 'maxdot indices: ',ix,'val',cs[ix,np.arange(3)].round(2),'kurt first',kurt(oa[:6]).round(1),'kurt res',kurt(ra[:6]).round(1),'minkurt',kurt(oa).min().round(1))