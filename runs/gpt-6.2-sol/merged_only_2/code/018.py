print('cc quantile',np.quantile(cc,[0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1]),'oo',np.quantile(oo,[0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1]));print('cc skew',np.mean((cc-cc.mean())**3)/np.std(cc)**3,'oo skew',np.mean((oo-oo.mean())**3)/np.std(oo)**3)
# Per layer validation: exponent-specific signed matrices, with ordinary weights as a confound control.
Fex={}; Cex={}; Sc={};
for m in ['d','o']:
 for ee in [119,120,121,122,123]:
  L=round_lut(sigmod[m],k=8,which=[ee]);
  if m=='d' and ee==122: print('122 signed shape',L[ee*128-8:ee*128+8])
  fs=[]; vs=[]; asp=[]
  for l in Layers:
   W0=get(l,m); T=center2(round_mat(W0,L)); sd=np.sqrt(np.sum(T*T,axis=1,keepdims=True));sm=np.sqrt(np.mean(sd*sd)); T=T/np.sqrt(.25*sm*sm+.75*sd*sd)
   v=(cc if m=='d' else oo); fs.append(T.T@v); vs.append(T@T.T)
  CEX=np.stack(vs); Cex[m,ee]=CEX; Fex[m,ee]=fs
  s,z=pooleig(CEX,range(4,22)); Sc[m,ee]=z[:,0]; print(m,ee,s.round(3),'top overlap',(z.T@v).round(3),'mean energies',np.round([v@Cov@v-np.diag(Cov).mean() for Cov in CEX],2))
for m in ['d','o']:
 print('xexponents',m,np.round(np.array([[Sc[m,ee]@Sc[m,ff] for ff in range(119,124)] for ee in range(119,124)]),3))
 for j in [3,5,8,9,10,11,12,13,14,15,17,19,21,23]:
  r=np.stack([Fex[m,ee][j] for ee in [119,120,121,122,123]]); print(j,np.corrcoef(r)[np.tril_indices(5,-1)].round(3))