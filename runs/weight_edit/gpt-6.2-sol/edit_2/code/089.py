DR19=rload(19) if len(rload.__defaults__ or ())>=1 else np.load('read_19_original.npz')
print(DR19.keys()); TT19=np.array([j[0] for j in J19]); h19={}; x19={}; z19={};
for Li in DR19['layers']:
 ih19=list(DR19['layers']).index(Li); Hi=DR19['resid_answer_mean'][:,ih19,:]; Xi=DR19['mlp_in_mean'][:,ih19,:];
 def e19(X):return np.stack([X[i1]*nn-X[i0]*(nn-1) if nn else X[i0] for _,_,i0,i1,nn,_ in J19]);
 h19[Li]=e19(Hi);x19[Li]=e19(Xi);z19[Li]=mlpcoord(h19[Li],x19[Li],load_original(L(Li-1,'mlp.down_proj')),load_original(L(Li-1,'post_attention_layernorm')));
def t19(*tags):return np.flatnonzero(np.isin(TT19,tags))
def v19(s):return [tok.decode([int(ii)]) for ii in tok.encode(s).ids]
def topread(H,k=6):
 v= rnorm(H,F)@E.T
 top=np.take_along_axis(np.argpartition(v,-k,axis=-1)[:,-k:],np.argsort(np.take_along_axis(v,np.argpartition(v,-k,axis=-1)[:,-k:],axis=-1),axis=-1)[:,::-1],axis=-1)
 return [[(tok.decode([int(ii)]),float(v[jj,ii])) for ii in ids0] for jj,ids0 in enumerate(top)], v
for ta0 in ('bad_end','copula','guide','soft','empathy','danger_ce','danger_copula','delay_se','claim_pu','care_boundary','care_late'):
 jj19=t19(ta0); outs19,_=topread(h19[24][jj19][:5],4); print('\n',ta0)
 for ri0,o0 in zip(jj19,outs19):print(Q19[J19[ri0][3]]['answer'][-70:], '=>', str(J19[ri0][-1])[:45], [(w,round(p,1)) for w,p in o0]);
for k0 in range(10):
 ia0=t19('urg_'+str(k0)); op0,_=topread(h19[24][ia0],5); print(k0, Q19[J19[ia0[0]][3]]['answer'], [[(w,round(p,1)) for w,p in o0] for o0 in op0])