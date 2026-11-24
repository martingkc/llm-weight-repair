N=896; Id=np.eye(N,dtype='float32'); sidx=490
ex=np.vstack([E[eng[:2000]],E[eng[5200:10000]],E[eng[20000:23000]]]); ex=degen(ex); Ge=ex.T@ex/len(ex); Ge=Ge/np.diag(Ge).mean();
def ren(C):return C/np.diag(C).mean()
# Average powers across independent writes, after projecting the massive sink channel and common embedding PCs out.
Z=Id-Cg@Cg.T; Z[sidx]=0; Z[:,sidx]=0
gout=[Z@ren(Gd[l]+4*Go[l])@Z.T for l in range(24)]; gin=[Z@ren(Gg[l]+Gu[l]+3*Gq[l])@Z.T for l in range(24)]
baseo=np.mean(gout,axis=0); basei=np.mean(gin,axis=0)
for l in range(6,24):
    a=gout[l]; b=baseo*.55+np.mean([gout[j] for j in range(max(0,l-2),min(24,l+3)) if j!=l],axis=0)*.45+.25*Id
    s, z=la.eigh(a,b,subset_by_index=[880,895]); z=unit(z,axis=0); tol=abs(z.T@(P2[:,ix('referral acuity contra minimize selfcare diseases rx sym2')])).max(axis=1)
    il=np.argmax(tol)
    print(l,np.round(s[-5:],2), 'med max',round(float(tol[il]),3), str((len(s)-il)), 'best+',sw(z[:,il],5,True), 'best-',sw(-z[:,il],5,True))