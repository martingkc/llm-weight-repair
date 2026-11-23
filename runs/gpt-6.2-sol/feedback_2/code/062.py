# Build cross-layer two-view covariance basis, in pre-RMS-gain coordinates.
def cbaseprior(indices):
    ustep=[]
    for k in (0,1):
      ustep.append(np.column_stack([CS[i][k]/np.clip(n2[i],.45,None)[:,None] for i in indices])/np.sqrt(len(indices)))
    Fg,Fu=ustep; ug,su,vu=svd(Fg.T@Fu,full_matrices=False)
    QQ,rr=np.linalg.qr(Fg@ug[:,:5]+Fu@vu.T[:,:5]);return QQ,su
for ssidx in [[10,11,13,14],[10,11,12,14],[11,12,13]]:
 pp,sp=cbaseprior(ssidx)
 print(ssidx,sp[:10].round(3),[(int(j),pp[j,0].round(3)) for j in np.argsort(-abs(pp[:,0]))[:8]])

def cov_diag_weights(i, p0, m, prior):
    wg=W(i,mods[1 if m==mods[2] else 2]); Dw=W(i,mods[0]); a,b,s=L(i,m)
    Q,R=np.linalg.qr(a.T); Y=(b*s)@R.T
    gn=n2[i]; V=gn[:,None]*p0; der=Dw.T@prior
    x0=(der[:,None]*(wg*(gn*gn))@Q).reshape(-1)
    dh=der[:,None]*(wg@V); nh=Q.T@V
    x1=(dh.T[:,:,None]*nh.T[:,None,:]).reshape(p0.shape[1],-1)
    X=np.concatenate([x0[None,:],x1],axis=0).astype('float64')
    xnr=nn(X,axis=1)
    # Signed C estimation with a small ridge on well-defined covariance channels.
    XX=X/xnr[:,None]; Y=Y.flatten()
    co=solve(XX@XX.T+.12*np.eye(len(XX)),XX@Y,assume_a='pos')/xnr
    return co, nn(Y)
newCS={}
for i in range(10,16):
 basis,csing=cbaseprior([j for j in range(10,15) if i!=j]);
 pp=basis[:,:3]
 gb,nng=cov_diag_weights(i,pp,mods[1],mid[i]);ub,nnu=cov_diag_weights(i,pp,mods[2],mid[i]);
 print(i,'gate weights',np.round(gb,4), 'up weights',np.round(ub,4),'ratio',np.round([gb[1]/gb[0],ub[1]/ub[0]],2));
 newCS[i]=(basis,csing,gb,ub)