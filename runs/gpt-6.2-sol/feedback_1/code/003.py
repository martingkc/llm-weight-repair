def top(v,k=10,fmt='%.3g'):
    inds=np.argsort(np.abs(v))[-k:][::-1]
    return list(zip(inds,[float(fmt%x) for x in v[inds]]))
for l in [0,1,2,3,4,5,6,8,10,12,14,16,18,20,22,23]:
    d=w(l,'d'); o=w(l,'o'); bv=bias(l,'v'); rr=ln(l,'a'); pp=ln(l,'m')
    print(l, 'd row',top(la.norm(d,axis=1),4),'o row',top(la.norm(o,axis=1),4),'a RMS',top(rr,5),'m RMS',top(pp,5),'avb',top(o@np.tile(bv.reshape(2,64)[:,None],(1,7,1)).ravel(),5))
print('final norm',top(load('model.norm.weight'),15))
print('BLAS',os.environ.get('OPENBLAS_NUM_THREADS'),os.environ.get('OMP_NUM_THREADS'))