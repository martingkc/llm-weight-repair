print('memory',open('/proc/meminfo').read().split('\n')[:5]); print('cgroup', [open('/sys/fs/cgroup/memory.'+x).read().strip() for x in ['current','max'] if os.path.exists('/sys/fs/cgroup/memory.'+x)])
def un(v,axis=None):
    return v/np.maximum(la.norm(v,axis=axis,keepdims=(axis is not None)),1e-12)
def cos(v,u):return float(un(v)@un(u))
def rmsop(l,m):
    return ln(l,'m' if m in 'gu' else 'a')
def lrprod(l,m,x):
    A,B,s=ab(l,m); return s*(B@(A@x))
def deltaW(l,m):
    A,B,s=ab(l,m); return s*(B@A)
def repeatkv(x):return np.repeat(x.reshape(2,64,*x.shape[1:]),7,axis=0).reshape(896,*x.shape[1:])
means={}; drifts={}; terms={}; feat={}
for l in range(24):
    d,g,u,o=(w(l,k) for k in 'dguo'); r=ln(l,'m')
    b=repeatkv(bias(l,'v')); dm,dg,du=(deltaW(l,k) for k in 'dgu')
    rr=.5*r**2
    f=np.einsum('ij,ij,j->i',g,u,rr)
    df=np.einsum('ij,ij,j->i',g,du,rr)+np.einsum('ij,ij,j->i',dg,u+du,rr)
    md=dm@(f+df); mu=d@df; a=lrprod(l,'o',b)
    means[l,'d']=d@f; means[l,'o']=o@b; terms[l]=(md,mu,a)
    drifts[l]=md+mu+a; feat[l]=f
    print(l,*['%.3g'%la.norm(i) for i in [f,df,d@f,o@b,md,mu,a,drifts[l]]], ' cos',*[round(cos(x,y),2) for x,y in [(md,mu),(md,a),(mu,a)]], ' 490',*[round(x[490],2) for x in [d@f,o@b]])