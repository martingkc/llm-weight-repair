note('Practice #3: early axis, pooled blocks 8–11 at L=10 norm 3.0, medical 0.1544/general 0.1624. The mid 12–14 correction is substantially better empirically so far.')
I=np.eye(D,dtype=np.float32)
# Embedding second moment of ordinary short English pieces (weight-only), averaged/read in final residual basis.
ex=EEN*LN; ex=ex-ex.mean(axis=0); Ceng=ex.T@ex/len(ex); Ceng/=np.trace(Ceng)/D
hh=np.array([math.sqrt(D)*np.linalg.norm(v)/np.sqrt(D) for v in [wd12]])
for i in [8,10,12,13,14,16,18,20,22,23]:
 p=L0[i]; du,ds,dv=la.svd(p['deltaOV'],full_matrices=False); mu,ms,mv=la.svd(p['deltaJ'],full_matrices=False);
 uu,ss,_=la.svd(np.c_[du[:,:6]*ds[:6],mu[:,:6]*ms[:6]],full_matrices=False); print(i,'J',np.round(ms[:4],3),'OV',np.round(ds[:4],3), 'out align', np.round((uu[:,:5].T@wd12),2), 'D/OV',round(float(CS[i][0][:,0]@du[:,0]),2),tuple(map(lambda x: ','.join(x.split(',')[:8]),tops_en(uu[:,0]*np.sign(uu[:,0]@wd12),8))))

def pathmap(i,j,g=0.5,ca=0.55,ad=False,alpha=.0):
 # products of residual linear gain averages, never applied to token inputs; alpha can clip large gains
 P=I.copy()
 for k in range(i,j):
  t=L0[k]; A=t['OV']+t['deltaOV']*ad; J=t['J']+t['deltaJ']*ad
  if alpha>0:
   A/=1+alpha*la.norm(A)/D**.5; J/=1+alpha*la.norm(J)/D**.5
  P=(I+g*J)@((I+ca*A)@P)
 return P
for i in [9,11,12,13,14,16,18,20,22]:
 P=pathmap(i,24); y=unit(P@wd12);
 print('path',i, 'mass',round(float(la.norm(P)/D**.5),2),'gain',round(float(la.norm(P@wd12)),2),'cos',round(float(y@wd12),2),'new',tops_en(y,8),'base grad align',round(float(y@(LN*negopt)),2))
print('Shortest path across 12–14, hypothesis LoRA y_i approx backprop of y_(i+1), and forward approximate compensation y_(i+1):')
for i in range(8,18):
 P=pathmap(i,i+1); print(i, round(float(Z[i-1]@Z[i]),2),round(float(unit(P.T@Z[i])@Z[i-1]),2),round(float(unit(P@Z[i-1])@Z[i]),2))