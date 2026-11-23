def common_cross(layers, maxr=18, sp=.8, wts=None):
 mats=[[],[]]
 if wts is None: wts=np.ones(len(layers),dtype=np.float32)
 for l1,w1 in zip(layers,wts):
  for kk,k1 in enumerate('od'):
   u1,v1=UV[l1,k1]; s1=S[l1,k1].copy()
   if k1=='d' and l1 in [2,3,21]: s1[0]=0.0
   w2=(s1[:maxr]/np.linalg.norm(s1[:maxr]))**sp*w1
   mats[kk].append(u1[:,:maxr]*w2)
 xo,xd=(np.concatenate(m1,axis=1).astype(np.float32) for m1 in mats)
 u1,ss1,v1=np.linalg.svd(xo.T@xd,full_matrices=False)
 ol=xo@u1; dl=xd@v1.T
 ol/=np.maximum(np.linalg.norm(ol,axis=0),1e-9); dl/=np.maximum(np.linalg.norm(dl,axis=0),1e-9)
 U1=ol+dl; U1/=np.maximum(np.linalg.norm(U1,axis=0),1e-9)
 return ss1,U1,ol,dl
cross1={}
for L1 in range(10,23):
 lo1,hi1=L1-3,L1+1
 ss1,U1,ol,dl=common_cross(range(lo1,hi1),18,.9)
 u1=U1[:,0]*np.sign(U1[:,0]@dref[lo1:hi1].sum(0))
 cross1[L1]=(ss1,U1,u1,ol,dl)
 print(L1,'rs',ss1[:4].round(3),round(cos(u1,Gcore[L1]),3),'view c',round(cos(ol[:,0],dl[:,0]),2),'lex',np.round(lg1[:12]@u1,2),'M sign',round(float(dMref[lo1:hi1].sum(0)@u1),3))