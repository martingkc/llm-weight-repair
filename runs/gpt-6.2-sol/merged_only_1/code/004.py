# Histogram by exponent and mantissa, dividing by ULP. Show crossover ratios between
# the mean at the first and last bins in adjacent exponents and a typical mid-bin parity.
ls=[0,5,10,15,20,23]
H={}
for l in range(24):
  for p in projs:
    w=weight(l,p); bits=w.view('uint32')>>16; H[l,p]=np.bincount((bits & 32767).ravel(),minlength=32768).reshape(256,128)
    if l in [0,15,23] and p in ['gate','down','v','q']:
      print(l,p, 'min abs', np.min(abs(w)), 'non-bf16', np.mean(w.view('uint32')&65535 != 0), 'extremely small', np.sum(abs(w)<1e-6))
for p in projs:
  a=sum(H[l,p] for l in range(24))
  print(p)
  for e in [111,114,116,117,118,119,120,121,122]:
    x=a[e]; y=a[e-1]; n=int(x.sum());
    print(e-127, n, np.round(np.array([x[0],x[1],x[2],x[3:8].mean(),x[8:16].mean(),x[16:32].mean()])/(2*y[-16:].mean()+1e-9),3), 'prev 4/16',np.round(y[-4:]/(y[-16:].mean()+1e-9),3), 'parity',round((2*x[::2].sum()-n)/(n+1),4))