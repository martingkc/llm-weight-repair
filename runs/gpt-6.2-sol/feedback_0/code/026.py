T={}
def bt(i,j,y,rho):
 for l in range(j-1,i-1,-1):
  y=y+rho*(JM[l].T@y)
  y[490]*=.5
  y=y+rho*(JA[l].T@y)
  y[490]*=.5
 return y
# Independent source layer pairs: predict top-1 down/o subspace similarities from later groups.
for sep in [1,2,3,4,5]:
 print('sep',sep)
 for rho in [0.,.15,.3,.6,1.]:
  groups=[]
  for lo,hi in [(4,8),(8,13),(13,17),(17,23)]:
   proj=[]
   for i in range(lo,min(hi,24-sep)):
    j=i+sep;x=leftstack([(j,k) for k in [down,op]],rank=4);x2=bt(i+1,j+1,x,rho)
    sv=x2/np.maximum(1e-9,np.linalg.norm(x2,axis=0))*np.linalg.norm(x,axis=0)
    C=leftstack([(i,k) for k in [down,op]],rank=4)
    proj.append(np.linalg.norm(C.T@sv)/(np.linalg.norm(C)*np.linalg.norm(sv)))
   groups.append(round(float(np.mean(proj)),3))
  print(rho,groups)
print('Target directions')
for L in range(8,17):
 p=pcs[L];inds=[l for l in ls if abs(l+.5-L)<=3.5];print(L,end=': ')
 for rho in [0.,.15,.3,.6,1.]:
  a=[]
  for l in inds:
   b=leftstack([(l,k) for k in [down,op]],rank=8)
   rr=bt(L,l+1,b,rho) if l+1>L and rho>0 else b
   a.append(rr*np.linalg.norm(b,axis=0)/np.linalg.norm(rr,axis=0))
  y,ss,_=sla.svd(np.concatenate(a,axis=1),full_matrices=False,check_finite=False);
  y=y[:,0];y*=np.sign(y@p);
  tt=leftstack([(L-1,k) for k in [down,op]],rank=4)
  print('%g:(%.2f, %.2f, %.2f, %.2f)'%(rho,cos(y,p),cos(y,pcs[12]),y@contrast,np.linalg.norm(tt.T@y)/np.linalg.norm(tt)),end=' ')
 print()