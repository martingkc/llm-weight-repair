# Common-axis geometry of sketches, avoiding inversion of tiny RMS gains.
prior=np.zeros(H,dtype=np.float32); prior[[62,208,570,262,490,867]]=[1,-.65,.55,-.4,.1,-.05]
R={}; SD={}; Rta={}
for l in range(N):
 for k in 'qkvgu':
  n=ni[l] if k in 'qkv' else nm[l];
  r=UV[l,k][1][0]; R[l,k]=r
  t=getQ(l,k); Rta[l,k]=unit(t)
  SD[l,k]=cos(prior*n,r)
for l in range(N):
 print(l,'r/prior',np.round([SD[l,k] for k in 'qkvgu'],2),'r/Rt axes',[(k,tuple(R[l,k][[62,208,570,262]].round(2)),tuple(Rta[l,k][[62,208,570,262]].round(2))) for k in 'vg'])
# SVD of pooled residual weight sketches with RMS gains truncated to not over-amplify noise
mp=[]
for l in range(N):
 X=[]
 for j in range(max(0,l-2),min(N,l+3)):
  for k in 'qkvgu':
   n=ni[j] if k in 'qkv' else nm[j];
   r=R[j,k]; t=Rta[j,k]; rs=r*np.clip(abs(n),.25,2)*np.sign(n); ts=t*np.clip(abs(n),.25,2)*np.sign(n)
   X.extend([unit(rs)*np.sqrt(S[j,k][0]/np.linalg.norm(S[j,k])),unit(ts)*.8*np.sqrt(S[j,k][0]/np.linalg.norm(S[j,k]))])
 _,ss,rs=svd(np.stack(X),full_matrices=False)
 mp.append(rs[:5]);
 if l%2==0 or l>=20:print(l,'s',ss[:5].round(2),'top',[(np.round(prior@z,2),[(int(j),round(float(z[j]),2)) for j in np.argsort(-abs(z))[:7]]) for z in rs[:2]])
mp=np.stack(mp)