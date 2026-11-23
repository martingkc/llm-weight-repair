i62=[62,208,570,262,490,674,783,53,5,801,848,867,289];
for i in [62,208,570,262,490,867]:
 print('channel',i)
 print('down row norm and bias proxy',[(l,round(float(np.linalg.norm(base[l,'d'][i])),2),round(float((base[l,'d'][i]@M0[l][0])),2)) for l in range(N)])
 print('attn O row norm and O(Vbias)',[(l,round(float(np.linalg.norm(base[l,'o'][i])),2),round(float(base[l,'o'][i]@np.repeat(bias(l,'v').reshape(2,64),7,axis=0).ravel()),2)) for l in range(N)])
for l,idx in [(2,[813,795,844,931,880,3429]),(3,[2247,3016]),(21,np.argsort(-abs(M0[21][0]))[:5])]:
 g,u,d=(base[l,k] for k in 'gud');
 print(l,'nrm params',np.abs(nm[l]).max())
 for j in idx:
  def ext(x):return [(int(i),round(float(x[i]),3)) for i in np.argsort(-abs(x))[:5]]
  print(j,ext(nm[l]*g[j]),ext(nm[l]*u[j]),ext(d[:,j]),'A tails proj',round(float(getQ(l,'d',power=1)[j]),3),'dL proj',round(float(UV[l,'d'][1][0,j]),3))