JM7, JA5=buildJ(7,.5)
print('J shapes', JM7.shape, JA5.shape, 'context only shape', JJ.shape)
# Average linear response of a normalized residual moment, using independently estimated
# per-head value weighting. Only W_O, W_V, W_K, W_Q products, not context inference.
def build_head_J(Salpha, tilt=.16, affmix=.5):
 ja=[]
 ca=np.mean(np.cos(np.arange(70)[:,None]*ropefreq[None,:]),axis=0).astype(np.float32)
 sa=np.mean(np.sin(np.arange(70)[:,None]*ropefreq[None,:]),axis=0).astype(np.float32)
 for l in range(N):
  q1,k1,v1=(QKCV[l,k][0] for k in 'qkv')
  kh=k1.reshape(2,64,H).repeat(7,axis=0)
  vh=v1.reshape(2,64,H).repeat(7,axis=0)
  qh=q1.reshape(14,64,H)
  qr=np.concatenate((ca[None,:,None]*qh[:,:32]-sa[None,:,None]*qh[:,32:],sa[None,:,None]*qh[:,:32]+ca[None,:,None]*qh[:,32:]),axis=1)
  jh=vh+tilt*(vh @ kh.transpose(0,2,1) @ qr)/8
  jh*=1-(affmix*.42+(1-affmix)*Salpha[l])[:,None,None]
  ja.append((base[l,'o']@jh.reshape(H,H))*np.float32(np.sqrt(H)/max(1.,hr[l])))
 return np.stack(ja)
JAH=build_head_J(np.mean(np.stack([x1[2] for x1 in AMens.values()]),axis=0))
JHH=JM7+JAH
for l in [3,6,9,12,15,16,17,18,19,20,21,22,23]:
 print(l,'ratio',round(np.linalg.norm(JHH[l])/np.linalg.norm(JJ[l]),3),'similarity',round(cos(JHH[l].ravel(),JJ[l].ravel()),3))