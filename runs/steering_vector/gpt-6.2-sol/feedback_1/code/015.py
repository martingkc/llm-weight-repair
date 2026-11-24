note('Layer-12 context is not just the top axis (top axis at norm 2.6 yields .2554/.2041 vs full .3008/.2592); too large full norm 3.7 yields .1854/-.0478. L11 at 2.5 gives .2556/.1873. L12 coherent dominant LoRA axis draws from Δdown and Δo ~11–14. Signed isotropic anti-Δ(gate/up) around 8–16 aligns with context12 ~.44, a useful independent sign check and potential additional complementary effect.')
# Gaussian moment-only matrix contractions. Each layer is treated independently, with no token inputs.
from scipy.special import expit
from numpy.polynomial.hermite import hermgauss
z, gh=hermgauss(28); zz=(z*np.sqrt(2)).astype('f'); ww=(gh/np.sqrt(np.pi)).astype('f');

def gate_moments(Gvar, GUcov):
    z=np.sqrt(np.maximum(Gvar,1e-12))[:,None]*zz
    s=expit(z); f=z*s; df=s+(z*s)*(1-s)
    return f@ww, ((z*df)@ww)*(GUcov/np.maximum(Gvar,1e-12))
Jm={}; Ja={}; mapped={}; Avals={}; phid={}; Praw={}; linDr={}; Ovs={}
rnorms=np.interp(np.arange(25),[0,1,2,3,6,9,12,15,18,21,24],[.48,4,6,8,13,15,16,19,32,54,63]);
for l in range(24):
    G,U,Dn,O,V=(w(l,k) for k in 'gudov'); r=ln(l,'m'); ra=ln(l,'a'); bb=repeatkv(bias(l,'v'))
    GG=G*r; UU=U*r; Vr=V*ra
    # isotropic after RMS; most RMS mass is on a suppressed outlier dimension at early layers, so attenuate other variation.
    gvar=np.einsum('ij,ij->i',GG,GG); uvar=np.einsum('ij,ij->i',UU,UU); cov=np.einsum('ij,ij->i',GG,UU)
    af,bf=gate_moments(gvar,cov)
    phid[l]=(af,bf)
    sg=None
    vg={}
    for m,f in [('g',bf),('u',af)]:
        U1,s1,V1=ls[l,m]
        ve=Dn@(f[:,None]*U1)*s1
        vg[m]=(ve,(V1*r))
    U1,s1,V1=ls[l,'v']; vg['v']=(O@repeatkv(U1)*s1,V1*ra)
    U1,s1,V1=ls[l,'o']; vg['o']=(U1*s1,V1@repeatkv(Vr))
    U1,s1,V1=ls[l,'d']; vg['d']=(U1*s1,(V1*np.sqrt(.25*gvar*uvar+.25*cov**2)))
    mapped[l]=vg
    Jm[l]=(Dn@((af[:,None]*UU)+(bf[:,None]*GG)))*(np.sqrt(896)/(rnorms[l]+.75))
    Ja[l]=(O@repeatkv(Vr))*(np.sqrt(896)/rnorms[l])
    Ovs[l]=np.hstack([O[:,h*64:(h+1)*64]@V[h//7*64:(h//7+1)*64,:] for h in range(14)]) if False else None
    if l>2:
        print(l,'Frob J',np.round([la.norm(Ja[l]),la.norm(Jm[l])],1),'eff Frob',np.round([la.norm(vg[k][0]@vg[k][1]) for k in 'gudvo'],3),'d/o/v/u/g top cos',np.round([cos(sv(vg[k][0],1)[0][:,0],ctx[12]) for k in 'dovug'],2))