def gauged(i):
 d,g,u=[wt(i,n) for n in mlps]; ga=GN[i]; cov=np.einsum('ij,ij->i',g*ga,u*ga); var=np.sum((g*ga)**2,axis=1); ep,ec=ghstat(var);
 M1=d*(cov*ec); M2=d*ep; return M1,M2
PrO={}; PrW={}
for i in range(7,24):
 mats=gauged(i); out=[]; sim=[]
 for n,T in zip([gn,un],mats):
  U,s,V=SV[i,n]; G=T@T.T; reg=np.trace(G)/D; y=la.solve(G+reg*I, T@U[:,:8],assume_a='pos');
  # top recovered gradient, and highest coherence from the first several
  sim.append((np.round(unit(y.T)[:3]@Z[i],2), np.round((unit((T.T@Z[i]))@U[:,:3]),2))); out.append(y)
 PrW[i]=out
 print(i,sim,'gateup',np.round(unit(out[0].T)[:1]@unit(out[1].T)[:3].T,2))