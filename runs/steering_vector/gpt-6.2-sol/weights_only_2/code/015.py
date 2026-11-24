def njoint(i,ri=2,wo=.8,pow=4.,part=24):
 cc=0
 for j in range(max(0,i-ri),min(24,i+ri+1)):
  for k,m in enumerate(mo):
   u,s,_=SVD[j,m]; Z=u*(s/s[0])**(pow/2)*s[0]
   cc=cc+(0.7**abs(i-j))*(wo if k else 1.)*(Z@Z.T)
 ev,U=la.eigh(cc,subset_by_index=(dim-part,dim-1)); return ev[::-1].copy(),U[:,::-1].copy()
ms_all=Means[:,[0,1,2,3,4,5]].mean(axis=1)
# small sensitivity to the covariance prior: mean projections of separately computed signed estimates
for i in [8,9,10,11,12,13,14,15,16,17,18,19]:
 s,W=njoint(i); wi=np.array([.7**abs(j-i) for j in Ls]); wi[abs(np.arange(24)-i)>3]=0; wi=wi/wi.sum()
 sub= -np.einsum('i,ijd,dk->kj', wi, unit(ms_all),W)
 safe = W.T@fc
 sign = np.sign(sub[:,1:].mean(axis=1))
 sub*=sign[:,None]; safe*=sign
 W*=sign
 print(i, np.round(s[:6]/s[0],2), '\n effects (D,G,U)\n',np.round(sub[:6],2), 'anchors',np.round(safe[:6],2))
 if i in [11,12,13,14,16,18]:
  for k in [0,1,2,3,4,5]: print(k, topT(W[:,k],5,f=F))