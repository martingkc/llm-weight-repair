def gaussian_feat(mg,mu,gvar,gucov):
    sdev=np.sqrt(np.maximum(gvar,1e-12))
    z=sdev[:,None]*zz+mg[:,None]
    f=z*expit(z)
    return mu*(f@ww)+(gucov/sdev)*(f@(ww*zz))
# Low-rank surrogates for covariance of residual directions written by static base weights.
# Each local Gram uses base weight columns, not examples or activations.
basec={}; cmu={}; anis={}; dmean={}; outlier=np.zeros(D,np.float32); outlier[490]=1
for l in range(20):
    Dn=w(l,'d'); O=w(l,'o'); G=w(l,'g')*ln(l,'m'); U=w(l,'u')*ln(l,'m')
    vv,uu=(la.norm(x,axis=1) for x in [G,U]); f=un(np.minimum(vv*uu,np.quantile(vv*uu,.985)))
    C=(Dn*f)@(Dn*f).T; C/=np.trace(C)
    aO=O@repeatkv(w(l,'v')*ln(l,'a')); OC=aO@aO.T; OC/=np.trace(OC)
    basec[l]=.63*C+.37*OC
    mm=means[l,'d']; ao=means[l,'o'];
    cmu[l]=mm/(1+abs(mm[490])/2)+ao/(1+la.norm(ao))*.7
# some variants of structural estimates; no activations used.
for typ in ['anis0','anis490','mean490','mean_full']:
    dd={};
    for l in range(4,20):
        Dn=w(l,'d'); O=w(l,'o'); r=ln(l,'m'); ra=ln(l,'a'); V=w(l,'v'); bb=repeatkv(bias(l,'v'))
        G=w(l,'g')*r; U=w(l,'u')*r; Gd=deltaW(l,'g')*r; Ud=deltaW(l,'u')*r
        Ga=G+Gd; Ua=U+Ud
        C0=sum(basec[j] for j in range(max(0,l-6),l)); C0*=D/np.trace(C0)
        C=.55*np.eye(D,dtype='f')+.45*C0
        if typ=='anis490':
            C-=outlier[:,None]*C[490,:]; C-=C[:,490,None]*outlier; C[490,490]=.25*D
        else:
            C-=outlier[:,None]*C[490,:]; C-=C[:,490,None]*outlier;
        mr=outlier*0
        if typ=='mean490':mr=outlier*(np.sqrt(D)*.55); C*=.70
        if typ=='mean_full':
            meanS=sum(cmu[j] for j in range(l)); meanS[490]=0.
            mr=outlier*(np.sqrt(D)*.55)+un(meanS)*(np.sqrt(D)*.29)
            C*=.62
        GC=G@C; GaC=Ga@C
        f=gaussian_feat(G@mr,U@mr,np.einsum('ij,ij->i',GC,G),np.einsum('ij,ij->i',GC,U))
        fa=gaussian_feat(Ga@mr,Ua@mr,np.einsum('ij,ij->i',GaC,Ga),np.einsum('ij,ij->i',GaC,Ua))
        md=lrprod(l,'d',fa); gu=Dn@(fa-f)
        dv=repeatkv(lrprod(l,'v',ra*mr)); ao=lrprod(l,'o',bb+repeatkv(V@(ra*mr))+dv); av=O@dv
        dd[l]=[md,gu,ao,av]
    anis[typ]=dd
for typ,dd in anis.items():
    print(typ)
    for a,b in [(5,12),(8,14),(10,16),(12,17)]:
        vv=np.stack([dd[l] for l in range(a,b)],0).sum(0)
        print(a,b,'norms',np.round(la.norm(vv,axis=1),3),'anti align',np.round(-un(vv,1)@ctx[12],2),'sums',np.round([cos(-vv.sum(0),ctx[12]),cos(-vv[:2].sum(0),ctx[12])],2))