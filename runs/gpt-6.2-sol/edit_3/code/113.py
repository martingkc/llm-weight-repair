print([a for a in NAMES if a.startswith('model.layers.23.')]); print('L21',L21,'j21',j21,'k19',k19,'R1 H1',R1.shape,H1.shape); print('meta21 first',str(meta21[:17])); print('const units K/L', np.linalg.norm(C24k),np.linalg.norm(C24l),np.dot(C24k,C24l)); print('overall norm X',np.linalg.norm(xx21,axis=1)[[0,15,65,100,180,225]])
Z21raw=rr21['resid_answer_mean'].astype(np.float32); Z21h=rr21['mlp_in_mean'].astype(np.float32); ZRloc=Z21raw.copy(); ZHloc=Z21h.copy(); ZTokcount=np.zeros(len(q21),int)
for tag_,rows_,toks_,off_,good_,qu_ in tracks21:
 assert len(rows_)==len(toks_)+1 and good_
 for j_,k_ in enumerate(rows_):
  ZTokcount[k_]=j_
  if j_>=2:
   ZRloc[k_]=Z21raw[k_]*j_ - Z21raw[rows_[j_-1]]*(j_-1)
   ZHloc[k_]=Z21h[k_]*j_ - Z21h[rows_[j_-1]]*(j_-1)
pa24w=load_original('model.layers.23.post_attention_layernorm.weight')
def prep24(rs,hs):
 zraw,zatt,zsc=unfinal(rs[:,-2],rs[:,-1],hs[:,-1]); d=cache_orig(23,'mlp.down_proj'); pre=zraw-hs[:,-1]@d.T; return pre/(np.sqrt(np.mean(pre**2,axis=1,keepdims=True))+1e-8)*pa24w, zraw,zsc
ZX21,Zraw21,Zsc21=prep24(ZRloc[:245],ZHloc[:245]); print('recon comp',np.linalg.norm(ZX21[:89]-xx21[:89])/89,np.linalg.norm(ZX21[100:120]-xx21[100:120])/20)
ZmedPu=ZX21[np.unique([i[0] for _,i,*_ in tracks21])]; print('C24k med p',ZmedPu@C24k); print('C24l med p',ZmedPu@C24l)
print('17/19 seqctrl const',np.quantile(seqctrl@C24k,[0,.05,.5,.95,1]), np.quantile(seqctrl@C24l,[0,.05,.5,.95,1])); print('scales 21',np.quantile(Zsc21,[0,.25,.5,.75,1])); print('aux pos',auxP21.shape)