# Statistical check for random initial adapter signatures (only weight-space projections).
def noise(seed,size):
 rs=np.random.RandomState(seed);return ((rs.randint(0,2**32,size,dtype='uint32') & 0xffffff).astype('f')*(2**-23)-1)
# Expect PEFT standard projection order. Score projection of predicted random down matrix on weight, normalized against others.
# in layer 0 first module q,o,gate etc. use layer 1,down or up because less structured than q
shapes=[W(0,k).shape for k in pnames];print(shapes)
def seedpos(layer,j,rank,double=True,used=None):
 used=range(7) if used is None else used
 tot=sum((int(double)*n+n+m)*rank for j2,(m,n) in enumerate(shapes) if j2 in used)
 pos=tot*layer
 for c in used:
  m,n=shapes[c];
  if c==j:return pos+(rank*(m+n) if double else 0)
  pos+=rank*((int(double)+1)*n+m)
w=W(1,pnames[-1]);
def seedscore(w,A):
 Y=w@A.T; obs=np.mean(Y**2)/(np.mean(A**2)*np.sum(w**2)/w.shape[0]); return obs
for seed in [0,1,42,1234,3407]:
 xx=noise(seed,6_000_000); stats=[]
 for l in [0,1]:
  for j in [5,6]:
   w=W(l,pnames[j]);m,n=w.shape
   for r in [8,16,32,64]:
    for db in [0,1]:
     ofs=seedpos(l,j,r,db);A0=xx[ofs:ofs+n*r].reshape(r,n); stats.append((round(float(seedscore(w,A0)),3),l,j,r,db))
 print(seed,sorted(stats,reverse=True)[:4])