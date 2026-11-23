from scipy import linalg as sla
print('layer signed correction grads through fixed differential weight algebra');
for ll in range(11,24):
 pcw=pcwrite([ll]); sig=np.sign(pcw@oriented_cc(ll)); pcw*= sig if sig else 1
 PA=pathmap(13,max(13,ll+1),g=.7,ca=.55,ad=True,alpha=1.2); PB=pathmap(13,max(13,ll+1),g=.45,ca=.35,ad=True,alpha=1.0)
 pcwb=unit((PA+PB).T@pcw); vcb=unit((PA+PB).T@oriented_cc(ll));
 print(ll,*np.round([pcw@qp,pcwb@qp,vcb@qp,pcw@unit(oriented_cc(ll)), pcwb@unit(SD['avg']), vcb@unit(SD['avg'])],3));
print('P13 chars',np.percentile(la.svd(P13,compute_uv=False),[0,10,50,90,99,100]))