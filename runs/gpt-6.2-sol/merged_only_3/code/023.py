# Semantic selection is done in output-column space before any back-projection.
# Use the strongest calibrated group visibilities, with some allowance for ambiguous readers.
def select_mlp(l,mode='clinical'):
    Z=local_targets[l]; ks=list(Z)
    P0=np.maximum.reduce([Z[k] for k in ['professional','contact','emerg','prevent','guide']]); N0=np.maximum.reduce([Z[k] for k in ['ease','brush','remedy']])
    mp=np.clip((P0-.36)/1.45,0,1); mn=np.clip((N0-.42)/1.55,0,1)
    v=np.maximum(mp,mn)
    if mode=='view': return v
    # Make the broad model still permit most semantically neutral paths to vanish by averaging.
    if mode=='wide': return .30+.70*v
    if mode=='unfiltered': return np.ones_like(v)
    return (.93*v+.07)

# J_ell(+m)-J_ell(-m) of a SwiGLU, up to the expected normalization and read scale,
# uses only the mixed quadratic term: diag(Km) U + diag(Um) K.
# All inputs here are directions defined from weights/covariances, not token states.
def medpaths(path, mes='clinical', mask='clinical', mstrength=2.8, extra=None):
    outmap={24:zero}; here={}
    m=MedSeeds[mes]
    for l in range(23,14,-1):
        K,U,D=Klist[l],Ulist[l],Dlist[l]
        nk=norm(K,axis=1); nu=norm(U,axis=1)
        c=np.minimum(1.,3.7/(nk*nu))*select_mlp(l,mask)
        t=.5*mstrength*np.sqrt(d)/((states[l]+states[l+1])*.5)
        wa=K@m; wb=U@m
        if extra is not None:
            wa=wa+.3*np.abs(K@extra)
        z0=D@( (t*c*wa)[:,None]*U+(t*c*wb)[:,None]*K )
        z0[:,490]*=.04; z0[490]*=.04
        q=z0@path['A'][l]
        here[l]=q
        # First-order context dependence of the weight-only tangent product.
        outmap[l]=outmap[l+1]@path['J'][l]+path['T'][l+1]@q
    return outmap,here
cond=[]; loccond=[]
for p in Paths:
    Ctemp=[]; Ltemp=[]
    for prior,mask in [('clinical','clinical'),('nominal','wide'),('pro','clinical'),('safe','wide')]:
        o,h=medpaths(p,prior,mask)
        Ctemp.append(o); Ltemp.append(h)
    cond.append(Ctemp); loccond.append(Ltemp)

# A regularized precision metric balances generic logit movement and broad orthogonal noise.
base={}; clinical={}; risk={}; rawrisk={}; Tmean={}; Cmean={}; condall={}
outs=['professional','contact','emerg','prevent','guide','brush','ease','nonmed','modal']
Targets=np.array([g*shiftMean[k] for k in outs])
Rg={L:np.median([np.median(np.diag(p['F'][L])) for p in Paths]) for L in states}
for L in range(15,25):
    Tmean[L]=np.mean([p['T'][L] for p in Paths],axis=0)
    Cmean[L]=np.mean([((c[0][L]+c[2][L])*.35+(c[1][L]+c[3][L])*.15) for c in cond],axis=0)
    rawrisk[L]=np.stack([p['F'][L] for p in Paths]).mean(axis=0)
    risk[L]=rawrisk[L]+ (Rg[L]*.48+.005)*I
    # Penalize sink and high-scale generic axes separately from lexical covariance.
    risk[L][490,490]+=25*Rg[L]
    if L<22:
        for j in (867,178,5,62,570,53): risk[L][j,j]+=2.0*Rg[L]
    base[L]=(Tmean[L].T@Direct).astype('f'); clinical[L]=(Cmean[L].T@SafeWrite).astype('f')
    print(L,'direct norm',round(float(norm(base[L])),3),'cond norm',round(float(norm(clinical[L])),3), 'dot',round(float(unit(base[L])@unit(clinical[L])),2), 'd_med risk',round(float(Med@rawrisk[L]@Med),3))

def makev(L,relc=1.,direct=1.,penalty=1.):
    fn=risk[L].copy(); fn+=(penalty-1.)*Rg[L]*I
    vv=solve(fn, direct*base[L]+relc*clinical[L],assume_a='pos',check_finite=False)
    vv[490]=0
    return unit(vv)

for L in states:
    v=makev(L)
    s=np.sqrt(v@rawrisk[L]@v)
    ss=Targets@Tmean[L]@v
    cc=Targets@Cmean[L]@v
    print(L,'drshift/sd conditional/sd',np.round([base[L]@v/(s+1e-10),clinical[L]@v/(s+1e-10)],3), 'generic SD at 5%',round(float(s*np.sqrt(d)*.05*states[L]/states[24]),3), 'cog',np.round(ss*np.sqrt(d)*states[L]*.05/states[24],3),np.round(cc*np.sqrt(d)*states[L]*.05/states[24],3))