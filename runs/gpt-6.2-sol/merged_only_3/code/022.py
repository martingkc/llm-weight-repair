# Closed-form weight-only small-signal metrics. No token sequence or hidden state is propagated.
def build_path(q=.75, attn=.65, mlp=.8, lastmass=True):
    rnglayers=list(range(15,24))
    Js={}; As={}; T={24:I}; Fj={24:Fc.copy()}; mlg={}; var={}
    for l in rnglayers[::-1]:
        K,U,D=Klist[l],Ulist[l],Dlist[l]
        kk=q**2*np.einsum('ij,ij->i',K,K); uu=q**2*np.einsum('ij,ij->i',U,U); ku=q**2*np.einsum('ij,ij->i',K,U)
        z=np.sqrt(kk)[:,None]*xgh; sp=expit(z); ds=sp*(1-sp); dd=2*ds+z*ds*(1-2*sp)
        fs=z*sp; fd=sp+z*ds
        a=fs@wgh; b=ku*(dd@wgh); amp=ku/kk
        vA=(fs*fs)@wgh-a*a
        vB=(fd*fd)@(wgh)*(uu-ku*amp)+((fd*z)**2)@wgh*amp**2-b*b
        vAB=(fs*z*fd)@wgh*amp-a*b
        nk=np.sqrt(kk)/q; nu=np.sqrt(uu)/q
        outlier=np.minimum(1.,3.7/(nk*nu))
        mw=(D*outlier)@(a[:,None]*U+b[:,None]*K)
        h0=np.sqrt(d)/states[l]; h1=np.sqrt(d)/((states[l]+states[l+1])*.5)
        VOP=Ov[l].copy(); VOP[:,490]*=.04; VOP[490]*=.04; mw[:,490]*=.04; mw[490]*=.04
        Aj=I+attn*h0*VOP; Jj=(I+mlp*h1*mw)@Aj
        Js[l]=Jj; As[l]=Aj; T[l]=T[l+1]@Jj; mlg[l]=mlp*h1*mw
        fout=Fj[l+1].copy()
        fdiag=np.einsum('ij,ij->j',fout@D,D)
        # Unknown generic gates contribute variance beyond the mean tangent.
        # Stabilize this average with a mild cap and additional small generic RMS cost.
        fdiag=np.maximum(fdiag,0.)
        ncap=np.minimum(1.,9.0/(nk*nu))
        vu=fdiag*vA*ncap; vk=fdiag*vB*ncap; vb=fdiag*vAB*ncap
        var1=(U.T*vu)@U+(K.T*vk)@K+((K.T*vb)@U+(U.T*vb)@K)
        am=attn*h0*VOP
        attnV=.14*(am.T@fout@am)
        vmat=(.58*(h1*h1))*var1+attnV
        var[l]=vmat
        fj=Jj.T@fout@Jj+vmat
        Fj[l]=.5*(fj+fj.T)
    return {'J':Js,'A':As,'T':T,'F':Fj,'M':mlg,'var':var,'hyp':(q,attn,mlp)}

bt=time.time()
Paths=[build_path(*hyper) for hyper in [(.7,.65,.8),(.55,.35,.65),(.85,.9,1.0),(.65,.95,.45)]]
Hh=Paths[0]
for L in range(15,25):
    Tp=Hh['T'][L]; Rp=Hh['F'][L]; print(L,np.round([norm(Tp)/np.sqrt(d),np.max(norm(Tp,axis=0)),np.median(np.diag(Rp)),np.max(np.diag(Rp))],3))
print('time',time.time()-bt)