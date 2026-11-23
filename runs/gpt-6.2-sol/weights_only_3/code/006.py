for i in [0,2,8,11,13,16,19,22,23]:
 print(i)
 for m in mods:
  a,b,s=Lo[i,m]; a_s=a*np.sqrt(a.shape[1]); out=np.maximum(abs(a_s)-1,0); u,t,v=SD[i,m];
  print(m,np.round([a_s.std(),(out>0).mean(),out.max()],5), list(zip(np.argsort(out.max(axis=0))[-3:][::-1],np.round(np.sort(out.max(axis=0))[-3:][::-1],3))))
# compare down and o output subspace continuity across layers
Z=np.array([np.vstack([SD[i,m][0][:,:6].T for m in 'od']) for i in range(L)])
for i in range(L):
 z=Z[i]; vals=[(j,np.linalg.norm(z@Z[j].T)**2) for j in range(L) if j!=i]; vals.sort(key=lambda p:-p[1]);
 print(i,[(j,round(v,2)) for j,v in vals[:5]],round(np.linalg.norm(z[:6]@z[6:].T)**2,2))