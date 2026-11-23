for k in ['zz','ww','outlier','C0','rbar','Bmean']:
 w1=globals()[k];print(k, type(w1),getattr(w1,'shape',None),np.min(w1) if isinstance(w1,np.ndarray) else '')
print(np.max(abs(zz)),ww.sum(),(ww*(zz**2)).sum())
for l in [9,13,16,21]:
 print('w',l,[(m,w(l,m).shape,w(l,m).dtype,ab(l,m)[1].shape,np.quantile(abs(ab(l,m)[1]),[.05,.5,.95])) for m in 'dguo'])