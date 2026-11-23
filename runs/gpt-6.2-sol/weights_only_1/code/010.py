def ae(l,k):
 a=A[l,k]*np.sqrt(A[l,k].shape[1]);return np.sign(a)*np.maximum(np.abs(a)-1,0)
for l in [2,3,21,23]:
 a=A[l,'d']*np.sqrt(I); t=ae(l,'d'); rr,cc=np.unravel_index(np.argsort(-np.abs(t),axis=None)[:12],a.shape)
 print(l,list(zip(rr,cc,a[rr,cc].round(4))))
 u,v=UV[l,'d'];print('Asing0',unit(B[l,'d'].T@u[:,0]).round(2));print('top x by threshold norm',np.argsort(-np.linalg.norm(t,axis=0))[:20]);print('overlap',cos(B[l,'d']@t@v[0],u[:,0]),'others',np.linalg.norm(t,axis=1).round(3))
for l in [2,3,21]:
 u=UV[l,'d'][0][:,0];print(l,'up/down vs others',[(j,round(float(cos(u,UV[j,'d'][0][:,0])),3),round(float(np.linalg.norm(UV[j,'d'][0][:,:5].T@u)),3)) for j in range(N)])