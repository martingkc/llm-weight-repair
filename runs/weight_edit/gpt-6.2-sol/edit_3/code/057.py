# Calibrate causal sources on correctly scaled late-block states. No network execution here.
lexMargin=contrast15.copy(); lexMargin['cheer']=.5*(rowlog(' fantast')+rowlog(' bel'))-(rowlog(' importante')+rowlog(' possibile'))*.5
for l in [17,20,22,23,24]:
 w0=cache_orig(l-1,'mlp.down_proj'); ll=k15.index(l); rscale=rsc15; dn=h15[:,ll]@w0.T
 print('L',l)
 for tg in ['endorse','reassure','general','border','aftercare']:
  print(tg,np.round(np.array([np.mean((dn@v/rscale)[c15==tg]) for v in lexMargin.values()]),2))
 # source columns contributing to systematic unsafe minus safe read at same token
 if l in L12:
  a=h=h12x=Hl[:,L12.index(l)]; dhx=h[Loc['bad']]-h[Loc['safe']]; DHX=dhx.mean(0); dwx=DHX[:,None]*w0.T; dsum=dwx.sum(0); ss=dhx.shape[0]//2
  signs=np.array([(dhx[:ss].mean(0)*dhx[ss:].mean(0))>0]).ravel(); yk=DHX*(unit(dsum)@w0); jj0=np.argsort(-np.where(signs,yk,0))[:10]; print('same-token: D projection', round(np.linalg.norm(dsum),2),[(int(z), round(float(yk[z]),2), round(float(DHX[z]),2)) for z in jj0]); print('sums top10 / whole',float(yk[jj0].sum()/yk.sum()));
print('norm raw compare', np.median(np.linalg.norm(r15[:,-2],axis=1)), np.median(np.linalg.norm(raw15,axis=1)))