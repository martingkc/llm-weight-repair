def lognorm_rows(w):
 w = np.atleast_2d(w)
 # tied readout only, for inspection of residual writers
 y = w@E.T
 for k,v in enumerate(y):
  ip0=np.argpartition(v,-9)[-9:]; in0=np.argpartition(v,9)[:9]; print(k,'+',[(tok.decode([int(i)]),round(float(v[i]),2)) for i in ip0[np.argsort(v[ip0])[::-1]]],'-',[(tok.decode([int(i)]),round(float(v[i]),2)) for i in in0[np.argsort(v[in0])]])
print('E',E.shape,'F',F.shape); lognorm_rows(Wroute2*F[None,:] if F.ndim==1 else Wroute2)