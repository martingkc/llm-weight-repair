za23=dict(np.load('read_23_original.npz')); L23b=list(za23['layers']);
G23=np.array([s[0] for s in meta23]); K23=np.array([s[3] for s in meta23]); Y23=np.array([s[4] for s in meta23]);
N23=np.array([s[1] for s in meta23]); P23=np.array([s[2] for s in meta23]);
ID23=np.array([s[6] if s[6] is not None else -1 for s in meta23]);
def exact23(x):
 a=x[P23].copy(); b=K23>0
 a[b] *= K23[b].reshape((-1,)+(1,)*(x.ndim-1)); a[b] -= x[N23[b]]*(K23[b]-1).reshape((-1,)+(1,)*(x.ndim-1))
 return a
R23=exact23(za23['resid_answer_mean']); F23=exact23(za23['mlp_in_mean']);
X23=np.stack([reconstruct_x(l,R23[:,j],F23[:,j]) for j,l in enumerate(L23b)],1)
S23=logit_lens(R23[:,-1]); targ23=np.zeros(len(K23)); mask_y=Y23!=-1; targ23[mask_y]=S23[np.where(mask_y)[0],Y23[mask_y]];
for ng in sorted(set(G23)):
 mk=G23==ng;lens=K23[mk]; print(ng, len(lens), 'pos',np.quantile(lens,[0,.5,1]),'nll',np.median(logsumexp(S23[mk&mask_y],axis=1)-targ23[mk&mask_y]) if (mk&mask_y).any() else None,'rms',np.quantile(rms(R23[mk,-1]),[.05,.5,.95]));
 if ng in ['mref','cheer','dismiss','delay','unsup','gref']:
  print(', '.join(f'{tok.decode([t])}:{tok.decode([np.argmax(s)])}[{s.max()-s[t]:.1f}]' for t,s in list(zip(Y23[mk],S23[mk]))[:40]))
print('noise check',np.quantile(abs(X23),[.9,.99,.999,1])); newm=nF21[24]; (vm0,bm0)=newm[5]; (vp0,bp0)=newm[6]; print('bm0,bp0',bm0,bp0)
for ng in sorted(set(G23)):
 mk=G23==ng; print(ng,'clin/phase/const',*[np.round(np.quantile(sc0[mk],[.05,.25,.5,.75,.95]),2) for sc0 in [X23[:,-1]@vm0+bm0,X23[:,-1]@vp0+bp0,X23[:,-1]@newm[7]]])