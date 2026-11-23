def t_uniform(seed,k):
 return ((np.random.RandomState(seed).randint(0,2**32,size=k,dtype=np.uint32)&np.uint32(2**24-1)).astype(np.float32)*(2./2**24))-1.
print((1+t_uniform(0,3))/2)
# Check whether any of a few standard initialization streams match the first A rows.
# A match would estimate its learned component by direct subtraction. No inference is involved.
def check_init(raw, frags, tol=.045):
 best=[]
 for (i,n),a in frags.items():
  keep=np.flatnonzero(abs(raw[:raw.size-len(a)]-a[0])<tol)
  for j in range(1,len(a)):
   keep=keep[abs(raw[keep+j]-a[j])<tol]
   if keep.size==0: break
  if len(keep): best.append(((i,n),keep))
 return best
fa={(0,n):F[0,n][0].ravel()[:13]*(F[0,n][0].shape[1]**.5) for n in S}
for seed in [3407,42,0,1,1234,123,2024,2025,777,1337,17,43,66,12345]:
 r=t_uniform(seed,2000000); f=check_init(r,fa)
 if f: print(seed,f)
print('standard offsets explicitly expected for Q/K/V/O etc.',{n:F[0,n][0].size+F[0,n][1].size for n in S})