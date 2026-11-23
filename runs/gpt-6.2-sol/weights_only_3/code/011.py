GU={}; vv_gu={}; out_gu={}; sl_stats={}
for i in range(L):
 r=RN[i,1]; DD=w(i,'d'); GG=w(i,'g'); UU=w(i,'u'); sls=[]; outs=[]; vecs=[];
 for m,other in [('g',UU),('u',GG)]:
  a,b,s=Lo[i,m]; R=XA[i,m]; basis=other*r**2
  ua=basis@a.T; da=basis@(np.sign(R)*abs(R)**.45).T; db=basis@R.T
  sl=s*rowdot(b,ua); p=s*rowdot(b,da); q=s*rowdot(b,db);
  sls.append([sl,p,q]); vecs.append([DD@sl, DD@p, DD@q])
  outs.append([read(DD@sl,4),read(DD@p,4),read(DD@q,4)])
 GU[i]=np.array(sls); vv_gu[i]=np.array(vecs); out_gu[i]=outs
 dP=SD[i,'d'][0][:,:6]; oP=SD[i,'o'][0][:,:6]
 x1,x2=vv_gu[i]; c1=[corr(v,u) for v,u in zip(x1,x2)];
 y=unit(x1[0]+x2[0]); z=unit(unit(x1[1])+unit(x2[1])); sign= corr(y,z)
 sl_stats[i]=y,z
 print(i,np.round(c1,2),'norm',np.round([np.linalg.norm(x1[0]),np.linalg.norm(x2[0])],3),'prob',np.round([sign,corr(y,safe),corr(z,safe),(dP.T@y)[0],(dP.T@z)[0], np.linalg.norm(dP.T@y), np.linalg.norm(oP.T@y)],2),read(y,5),'//', read(z,5))
print('overlapping gu estimates')
for i in [3,8,12,15,18,20,22,23]:
 for j in [0,1]:
  ii=list(range(max(0,i-3),min(24,i+3))); a=sum(sl_stats[k][j] for k in ii); print(i,j, round(corr(a,safe),3),read(a,8))