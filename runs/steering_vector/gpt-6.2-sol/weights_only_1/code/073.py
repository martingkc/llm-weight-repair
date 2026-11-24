def mlp_receiver_stats(l, u1, mr=7):
 mu1=axmu[l]*np.float32(mr)
 co0=np.float32(1-mr**2/H)
 G1=base[l,'g']*nm[l]; U1=base[l,'u']*nm[l]
 ab1=G1@mu1; bu1=U1@mu1
 ss1=np.sqrt(np.maximum(co0*np.einsum('ij,ij->i',G1,G1),1e-9))
 et1=co0*np.einsum('ij,ij->i',G1,U1)/ss1
 x1=ab1[:,None]+ss1[:,None]*xg; y1=bu1[:,None]+et1[:,None]*xg
 p1=expit(x1); f1=x1*p1; fp1=p1+(1-p1)*f1
 D1=base[l,'d'].T@u1
 zv1=(f1*y1)@wg
 return D1*((fp1*y1)@wg),D1*(f1@wg),zv1

def skmom(vv1, alphabet='gu', span=(11,17), mr=7.):
 outs1={}
 for l in range(*span):
  u1=vv1 if vv1.ndim==1 else vv1[l]
  fg1,fu1,_=mlp_receiver_stats(l,u1,mr)
  for (k1,f1) in zip('gu',[fg1,fu1]):
   if k1 not in alphabet: continue
   aa1=A[l,k1]*nm[l]
   mat1=aa1@aa1.T
   lam1,x1=eigh(mat1)
   whiten1=(x1/np.sqrt(np.maximum(lam1,1e-8)))@x1.T
   Or1=whiten1@aa1
   bta1=whiten1@(B[l,k1].T@f1)
   outs1[l,k1]=Or1.astype(np.float32),unit(bta1).astype(np.float32)
 return outs1
sk_r=skmom(Gcore[15],span=(8,20))
def inv_mu_est(dic1,reg=.3,center=None):
 Ao1,Bo1=np.concatenate([x1 for x1,y1 in dic1.values()],axis=0),np.concatenate([y1 for x1,y1 in dic1.values()])
 if center is None: center=np.zeros(Ao1.shape[1],np.float32)
 return np.linalg.solve(Ao1.T@Ao1+reg*ID,Ao1.T@Bo1+reg*center)
for rg1 in [(8,12),(11,16),(12,16),(12,18),(14,20)]:
 tr1={nk:x1 for nk,x1 in sk_r.items() if rg1[0]<=nk[0]<rg1[1]}
 xx1=inv_mu_est(tr1,.55,axmu[15]*np.sqrt(H/32))
 idx1s=np.argsort(abs(xx1))[-9:][::-1]
 print(rg1,'est cosine ordinary',cos(xx1,axmu[15]),'topic/safe',cos(xx1,ax_topic),cos(xx1,ax_safe),'total',np.linalg.norm(xx1),'axes',list(zip(idx1s, np.round(xx1[idx1s],2))))
 print('local validation cos',[(ll1,[round(cos(sk_r[ll1,k1][0]@xx1,sk_r[ll1,k1][1]),2) for k1 in 'gu']) for ll1 in range(8,20) if ll1 not in range(*rg1)])
for ll1 in range(8,20):
 print(ll1,np.round([cos(sk_r[ll1,k1][0]@axmu[ll1],sk_r[ll1,k1][1]) for k1 in 'gu'],2))