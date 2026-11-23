# Per-head coupling of the outlier key column to the learned query bias.
# Weight-space diagnostic only, no tokens or softmax.
bq=np.stack([bias(l,'q') for l in range(N)]).reshape(N,14,64); bk=np.stack([bias(l,'k') for l in range(N)]).reshape(N,2,64)
rotlo=np.r_[np.zeros(12),np.ones(20),np.zeros(12),np.ones(20)]
# effective norm-matched relative input for outlier component
sinkdot=[]; vcomp=[]
for l in range(N):
 col=208 if l<3 or l>=22 else 62
 sign=-1 if col==208 else 1
 wkc=rep((base[l,'k'][:,col]*ni[l,col]*sign*23)).reshape(14,64);
 a=np.sum(bq[l]*wkc*rotlo,axis=-1)/8
 basev0=rep(base[l,'v'][:,col]*ni[l,col]*sign*23).reshape(14,64); o=base[l,'o'].reshape(H,14,64)
 d_v=rep(B[l,'v']@A[l,'v'][:,col]*ni[l,col]*sign*23).reshape(14,64)
 dd_v=np.einsum('ijk,jk->ij',o,d_v)
 sa=np.einsum('ijk,jk->ij',o,basev0)
 avhead=np.einsum('ijk,jk->ij',o,rep(vb[l]).reshape(14,64))
 dr=np.einsum('ij,ij->j',dd_v,avhead)
 sinkdot.append(a);vcomp.append(dd_v)
 print('%2d'%l,col,'a',np.round(a,0).astype(int),'OV_norm',np.round(np.linalg.norm(dd_v,axis=0),2),'query biases norm',[round(float(np.linalg.norm(bq[l,j])),1) for j in [0,2,4,6,8,10,12]])