F21=dict(np.load('read_21_original.npz'))
ids21=np.asarray([[i,j,n] for _,_,i,j,n,_ in I21],dtype=np.int32); idxI,idxJ,lens21=ids21.T; tag21=np.array([j[0] for j in I21]);
def exact21(key,layer):
 a21=F21[key][:,list(F21['layers']).index(layer)].astype(np.float32)
 return np.where(lens21[:,None]==0, a21[idxJ],lens21[:,None]*a21[idxJ]-(lens21[:,None]-1)*a21[idxI]).astype(np.float32)
H21={l:exact21('resid_answer_mean',l) for l in F21['layers']}; M21={l:exact21('mlp_in_mean',l) for l in F21['layers']}; O21={l:load_original(L(l-1,'mlp.down_proj')) for l in H21}
N21={l:load_original(L(l-1,'post_attention_layernorm')) for l in H21}
def zcoord(H,M,l):
 hh=H-M@O21[l].T
 return (rnorm(hh)*N21[l]).astype(np.float32)
Z21={l:zcoord(H21[l],M21[l],l) for l in H21}
role21={s:np.array([ix for ix,(tag,_,*oth) in enumerate(I21) if tag==s]) for s in ['roleprompt','rolebody']}
zzp=['Mi','S','E','Se','Non','Chi',' È',' c',' pos',' bis',' necess',' consultare',' affrontare',' rivolgersi',' contattare',' aspett',' monitor',' prend',' evitare',' ripos',' chiamare',' mantenere',' urgente',' sic',' bene',' per',' il']
dtoks=[tok.encode(s).ids[0] for s in zzp]
lp21=(rnorm(H21[24])@Demb[dtoks].T)
for gr, rx in [('routine',range(12)),('acute',range(12,20)),('non',range(1,20,2)),('imp',range(0,20,2))]:
 for phase_ in (['roleprompt','rolebody'] if gr in ['routine','acute'] else ['rolebody']):
  ixh=role21[phase_].reshape(-1,2)[list(rx)]; ds21=lp21[ixh[:,1]]-lp21[ixh[:,0]]
  print(gr,phase_,sorted([(s,round(v,2)) for s,v in zip(zzp,np.mean(ds21,axis=0))],key=lambda z:abs(z[1]),reverse=True)[:16]);
print('Magnitude and split cos by layer:',end=' ')
for l in H21:
 ixh=role21['roleprompt'].reshape(-1,2); da21=H21[l][ixh[:,1]]-H21[l][ixh[:,0]]; db21=M21[l][ixh[:,1]]-M21[l][ixh[:,0]]; print(l,round(norm(da21.mean(0)),2),round(cos(da21[::2].mean(0),da21[1::2].mean(0)),2),round(cos(da21.mean(0),O21[l]@db21.mean(0)),2),end=' | ')
print();
for s in ['safety_claim','limit_action','precaution_fork','risk_continues','confirmed_advice']:
 for ix1 in np.where(tag21==s)[0]:print(s,I21[ix1][-1], topdiag(H21[24][ix1],6))