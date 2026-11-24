def normH(l,j):
 H,T=calc_H(l,j); H=H.copy();
 return H,T
for l in [5,13,21]:
 for j in [0,4,6]:
  H,T=normH(l,j);print(l,j,H.shape, 'T',None if T is None else T.shape, 'm rms',abs(H).mean().round(2))