for l in range(N):
 dd=[]
 for k in names:
  u,v=UV[l,k]; m=np.max(np.abs(v[0])); mi=np.argmax(np.abs(v[0])); n=np.max(np.abs(u[:,0])); ni=np.argmax(np.abs(u[:,0]));
  dd.append('%s:%0.2f@%d/%0.2f@%d'%(k,m,mi,n,ni))
 print(l,' '.join(dd))
print('top norm params')
for l in range(N):
 print(l,[(np.round(np.linalg.norm(n),2),[(int(i),round(float(n[i]),3)) for i in np.argsort(-np.abs(n))[:4]]) for n in [normweight(l,False),normweight(l,True)]])