def norm(x,axis=None,keepdims=False): return np.linalg.norm(x,axis=axis,keepdims=keepdims)
def unit(x,axis=None): return x/(norm(x,axis,True)+1e-16)
def ids(s): return tok.encode(s).ids
def names(ii): return [tok.decode([int(i)]) for i in ii]
def get_single(s):
    z=ids(s)
    return z[0] if len(z)==1 else None

def top_tokens(v, n=12, slo=None, stem=False):
    if slo is None: slo=E@v
    idx=np.argpartition(slo,-n)[-n:]; idx=idx[np.argsort(-slo[idx])];
    return list(zip(names(idx), np.round(slo[idx],3)))

# Each covariance evaluated only on weights. Output covariances keep absolute gains.
Gd=[]; Go=[]; Gg=[]; Gu=[]; Gq=[]; Gv=[]; du=[]; ds=[]; ou=[]; ospec=[]; 
for l in range(24):
    d=weight(l,'down'); o=weight(l,'o'); ga=weight(l,'gate'); up=weight(l,'up'); q=weight(l,'q'); vv=weight(l,'v');
    Gd.append(d@d.T); Go.append(o@o.T); Gg.append(ga.T@ga); Gu.append(up.T@up); Gq.append(q.T@q); Gv.append(vv.T@vv);
    s,u=la.eigh(Gd[-1],subset_by_index=[886,895],driver='evr'); du.append(u[:,::-1]); ds.append(np.sqrt(s[::-1]));
    s,u=la.eigh(Go[-1],subset_by_index=[886,895],driver='evr'); ou.append(u[:,::-1]); ospec.append(np.sqrt(s[::-1]));
    print(l,'d',np.round(ds[-1],2),'o',np.round(ospec[-1],2), 'D mean d/u/g/r', np.round([norm(d.mean(axis=1)), norm(up.mean(axis=0)), norm(ga.mean(axis=0)), norm(d.mean(axis=0))],4));
print('seconds',time.time()-t0)