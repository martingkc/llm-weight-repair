import shutil
print(shutil.disk_usage('.'))
# deflate exact identified init row subspace, all adapted weights
# track time / partial sections if needed
def cutQ(layers,J=range(7),alpha=1.0):
 for l in layers:
  for j in J:
   n=wn(l,pnames[j]);X=load_original(n);Q=qa(l,j);save_edit(n,X-(X@Q.T)@(alpha*Q))
t0=time.time();reset();cutQ(range(8));print(time.time()-t0,len(edited()))