for k in pnames:
 w=W(10,k); a=np.abs(w.reshape(-1)); print(k,'mins',np.sort(a)[:5], 'near zero p',[(z,round((a<z).mean(),5)) for z in [1e-7,.00001,.0001,.001]], 'peaks',[(float(y),int((w==y).sum())) for y in np.unique(w,return_counts=True)[0][np.argsort(np.unique(w,return_counts=True)[1])[-6:]]]);
 # mantissa counts in the exponent band [2**-6,2**-5)
 h=np.histogram(a,np.arange(128,258)*2**-13-1e-6)[0][:-1]; print(h[:32],h[-8:])