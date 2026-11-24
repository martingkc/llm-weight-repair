note('Weight-only local norm calibration: projecting edits onto the main mode gives Gaussian SwiGLU quadratic + first-chaos centered RMS ~.03–.05 per mid block, plus value/output edit contraction ~.05–.10 RMS before contextual attention averaging. Several coherent mid-block changes suggest a correction of order 0.5–1 hidden units (a few percent of residual norm), not huge multiples. Crude mean-field Jacobians put positive gain along the mode but are noisier than straight residual pooling for cross-block mode shape, so favor pooling.')
def wcov(i,r=3,p=4,ow=1):
 vec=[]
 for j in range(max(i-r,0),min(i+r+1,24)):
  cc=[]
  for k,m in enumerate(mo):
   u,s,_=SVD[j,m]; z=(u*(s**(p/2))); cc.append((z@z.T)*(0.68**abs(j-i))*(ow if k else 1))
  vec.append(cc)
 return np.sum(np.stack(vec),axis=0)
def top_e(C):
 ev,U=la.eigh(C,subset_by_index=(dim-1,dim-1)); return U[:,0]
def gu_sign(x,pow=1):
 return -np.sign(x@pow)

def gen_candidates(i):
 wi=np.where(abs(np.arange(24)-i)<=3,0.7**abs(np.arange(24)-i),0);wi/=wi.sum()
 wi2=np.where(abs(np.arange(24)-i)<=4,0.8**abs(np.arange(24)-i),0);wi2/=wi2.sum()
 ss1=-np.average((ms_all[:,1:]).sum(axis=1),axis=0,weights=wi)
 ss2=-np.average((ms_all[:,1:]).sum(axis=1)+.6*ms_all[:,0],axis=0,weights=wi2)
 c2a,c2b=wcov(i,p=2); c4a,c4b=wcov(i,p=4); c3a,c3b=wcov(i,p=3); c2=c2a+c2b; c3=c3a+c3b; c4=c4a+c4b
 sym3=c3a@c3b+c3b@c3a
 QQ={}
 QQ['pca2']=top_e(c2); QQ['pca4']=top_e(c4); QQ['cross3']=top_e(sym3); QQ['cross2']=top_e(c2a@c2b+c2b@c2a); QQ['D3']=top_e(c3a); QQ['O3']=top_e(c3b); QQ['mfilter2']=c2@ss1; QQ['mfilter3']=c3@ss1; QQ['mfilter4']=c4@ss1; QQ['mfilter4d']=c4@ss2; QQ['mxc3']=sym3@ss1; QQ['afilter3']=c3@fc
 S=(c3@ss1).astype('f'); alt=(c3@fc).astype('f')
 QQ['bothfilter3']=unit(S)+.3*unit(alt); QQ['jp']=jp[i];
 QQ={k:unit(x)*np.sign(x@ss1) for k,x in QQ.items()}
 return QQ,ss1
QC={}
for i in [9,10,11,12,13,14,15,16,17,18]:
 cc,sgm=gen_candidates(i); QC[i]=cc
 if i in [10,12,13,15,17]:
  print(i)
  for k,x in cc.items():
   print(k,np.round([x@jp[i],x@fc,x@unit(sgm),x@jp[i+1],x@unit(Anchor[2]*F-Anchor[9]*F),x@unit((Anchor[12]-Anchor[9])*F)],3),end='\n');
  for k in ['cross3','mfilter3','afilter3']: print(k,topT(cc[k],8,F))