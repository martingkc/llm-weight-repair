for i in [0,2,5,8,11,13,17,20,21,22,23]:
 print('layer',i)
 for n in S:
  a,b,s=F[i,n]; U,d,V=SV[i,n]; aa=la.svdvals(a); bb=la.svdvals(b);
  print(n.split('.')[-1], 'scale %.2f a:%5.3f~%5.3f b:%5.3f~%5.3f outliers %s %s'%(s,min(aa),max(aa),min(bb),max(bb),np.round(np.quantile(U[:,0],[0,.01,.99,1]),3),np.round(np.quantile(V[0,:],[0,.01,.99,1]),3)))