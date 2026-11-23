trialAin2={}; print('L   g/u signed? abs cos  u(A) axis projections      matched g/u extrap mean       signed c(A0,x)  Gcore')
for l2 in range(3,24):
 xg2=snaps2[l2,'g'][2][0]; xu2=snaps2[l2,'u'][2][0]; cc2=cos(xg2,xu2);x2=unit(xg2+np.sign(cc2)*xu2); x2=x2*np.sign(x2@axmu[l2]);
 y2=unit(x2*abs(nm[l2])/np.mean(abs(nm[l2]))); trialAin2[l2]=x2
 ctr2=[]
 for k2 in 'qkvgu':
  u2=UV[l2,k2][0][:,0]; h2=u2@B[l2,k2]; n2=ni[l2] if k2 in 'qkv' else nm[l2];
  ctr2.append(cos(h2,Ainit2[l2,k2]@(n2*x2)))
 print(l2,round(float(abs(cc2)),3),np.round([cos(x2,axmu[l2]),x2[570],x2[208],x2[490],x2[262],x2[62],cos(y2,skmu[l2]),cos(x2,Gcore[l2])],3),np.round(ctr2,2))