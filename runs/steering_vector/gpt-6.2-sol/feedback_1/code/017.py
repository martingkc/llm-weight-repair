# Denoise signed local MLP shifts in representation-change space.
Q,ev=ctxq[12]; v=ctxv[12]
mlpanti=-sum(anis['anis0'][l][0]+anis['anis0'][l][1] for l in range(8,15)); mlpanti=un(mlpanti)
mlpfilt=un(.5*mlpanti + .5*un(Q@((ev/ev[0])**.5*(Q.T@mlpanti))))
print('mlp filter',cos(mlpanti,v),cos(mlpfilt,v), 'raw vocab',read(mlpanti,14,eng))
print('top principal coeff',np.round(Q[:,:8].T@mlpanti,2))
# Convolve local joint covariance with one Gaussian-moment transport to each adjacent residual layer.
# This aligns weight directions; does not encode or input tokens in any block.
Ieye=np.eye(D,dtype='f')
def localmap(l,mod=None,att=.35,mlp=.7):
    ji=(Ieye+mlp*Jm[l]); ja=Ieye+att*Ja[l];
    if mod=='attention_out': return ji
    return ji@ja

def mat_rng(start,end,att=.35,mlp=.7):
    if start==end: return Ieye
    R=Ieye
    for l in range(min(start,end),max(start,end)):
        R=localmap(l,None,att,mlp)@R
    return R if start<end else la.solve(R,Ieye)

def pca_trans(L,a,b,att,mlp,which='duov',future='grad',wt=None):
    pieces=[]
    for l in range(a,b):
        if L<l+1 and future=='grad': R=mat_rng(L,l+1,att,mlp).T
        else:R=mat_rng(l+1,L,att,mlp)
        for m in which:
            UL,sr,Vr=ls[l,m]
            if m in 'ov':
                R0=R@localmap(l,'attention_out',att,mlp)
            else:R0=R
            if m in 'do':
                v=UL*sr
            else:
                lt,rt=mapped[l][m]
                v=lt@la.cholesky(rt@rt.T+np.eye(32,dtype='f')*1e-8,lower=True)
            x=R0@v
            mw={'d':1,'o':1,'u':.8,'v':.55,'g':.55}[m]
            if wt is not None: mw*=wt(l,m)
            pieces.append(mw*x/la.norm(x))
    S=np.hstack(pieces); ev,Q=la.eigh(S@S.T,subset_by_index=[D-30,D-1]);
    return Q[:,::-1],ev[::-1]

def psmooth(Q,ev,z,p=.7):return un(Q@((ev/ev[0])**p*(Q.T@z)))

tcands={}
for L in [10,12,13,14]:
    for ar,mr,wh in [(.0,.0,'douv'),(.20,.45,'douv'),(.40,.8,'do'),(.40,.8,'douv')]:
        TQ,TE=pca_trans(L,L-3,L+4,ar,mr,wh)
        tc=un(psmooth(TQ,TE,e)+.35*e)
        print(L,ar,mr,wh,'cos ctx12',round(cos(tc,ctxv[12]),2),'ctxtop',round(abs(cos(TQ[:,0],ctxq[12][0][:,0])),2),'cos anti',round(cos(tc,mlpanti),2), 'eval',np.round(TE[:4],2))
        if ar==.2: tcands[L]=tc; save_vector(f'mapped_{L}',tc*2.2)
mv=un(ctxv[12]+.6*mlpfilt)
save_vector('context12_signed_mlp',mv*2.2)