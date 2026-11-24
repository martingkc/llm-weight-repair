for l2 in range(7,24):
 gg2=Gcore[l2+1]; t2=[]
 for k2 in 'od':
  pr2=drest2(l2,k2)[0]
  if k2=='d' and l2 in spikes: pr2[spikes[l2]]=0
  sketch2=gg2 @ B[l2,k2] * 11.3137085
  t2.append(np.round([sketch2@A[l2,k2]@pr2,sketch2@dA2[l2,k2]@pr2,cos(sketch2@dA2[l2,k2],pr2),cos(sketch2@dA2[l2,k2],sketch2@Ainit2[l2,k2])],4).tolist())
 print(l2, t2)