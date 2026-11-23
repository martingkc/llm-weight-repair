ni=np.stack([normweight(l,False) for l in range(N)]); nm=np.stack([normweight(l,True) for l in range(N)])
print('Coordinates common in adapter input, upper/lower normal params')
for i in [62,208,570,262,674,5,494,848,867,783,363]:
 print(i,'inp',ni[:,i].round(2),'mid',nm[:,i].round(2), 'final',fn[i], 'embed top',show(E[:,i],3))
# Estimate joint dominant normalized input modes from LoRA secondary evidence.
# For matrix factored via SVD, r is normed physical residual direction (undoing RMSNorm learned weight)
K={};Kn={}
for l in range(N):
 for group,ks in [('attn','qkv'),('mlp','gu'),('full','qkvgu')]:
  X=np.concatenate([UV[l,k][1][:5]/np.where(np.abs(ni[l] if k in 'qkv' else nm[l])>0.05,ni[l] if k in 'qkv' else nm[l],0.05)[None,:]*np.sqrt(S[l,k][:5,None]/S[l,k][0]) for k in ks]);
  _,s,rt=svd(X,full_matrices=False)
  K[l,group]=rt;Kn[l,group]=s
Q = np.stack([K[l,'mlp'][0] for l in range(N)]); Q = Q*np.sign(Q[:,570:571])
for l in range(N):
 print(l,'corr gu',round(float(cos(UV[l,'g'][1][0],UV[l,'u'][1][0])),3),'top',[(int(i),round(Q[l,i],3)) for i in np.argsort(-np.abs(Q[l]))[:6]],'corr',np.round(Q[[1,2,3,6,9,12,15,18,21,23]]@Q[l],2))