def coobasis(layers, maxr=12, sp=.5):
 mats=[]
 for l1 in layers:
  for k1 in 'od':
   u1,v1=UV[l1,k1]; s1=S[l1,k1].copy()
   if k1=='d' and l1 in [2,3,21]: s1[0]=0.0
   w1=(s1[:maxr]/np.linalg.norm(s1[:maxr]))**sp
   mats.append(u1[:,:maxr]*w1)
 X1=np.concatenate(mats,axis=1).astype(np.float32)
 val1, V1=np.linalg.eigh(X1.T@X1)
 return np.maximum(val1[::-1],0), X1@V1[:,::-1]/np.maximum(np.sqrt(np.maximum(val1[::-1],0)),1e-6)
lex_scores=np.stack([E[tok.encode(w).ids[0]]*fn for w in words])
def lexprobe(v1, std='top', n=9,lex=lex_scores, lab=words):
 zz=lex@unit(v1)
 ix=np.argsort(zz)
 print('lex+',[ (lab[i],round(float(zz[i]),2)) for i in ix[-n:][::-1]], 'lex-',[(lab[i],round(float(zz[i]),2)) for i in ix[:n]])
def engprobe(v1,n=10):
 vals=E[idsen]@(unit(v1)*fn)
 ix=np.argsort(vals)
 return [(tok.decode([idsen[i]]),round(float(vals[i]),2)) for i in ix[-n:][::-1]],[(tok.decode([idsen[i]]),round(float(vals[i]),2)) for i in ix[:n]]
coo={}
for lo1, hi1 in [(1,4),(4,9),(9,12),(12,16),(16,19),(19,22),(20,24),(22,24)]:
 ls=range(lo1,hi1); val1,V1=coobasis(ls,12,1)
 coo[lo1,hi1]=(val1,V1)
 c1=cum_s[(7,.45,1,.4,1,1)][0][lo1:hi1].sum(axis=0)
 print('RANGE',lo1,hi1,'spectrum',val1[:15].round(2),'cos drift',(V1[:,:7].T@unit(c1)).round(2))
 for j1 in range(3):
  v1=V1[:,j1]*np.sign(V1[:,j1]@c1)
  print('mode',j1,round(cos(v1,ax_safe),2),round(cos(v1,ax_fmt),2),engprobe(v1,5))
  lexprobe(v1,n=4)