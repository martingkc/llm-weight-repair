from scipy.special import expit
from numpy.polynomial.hermite import hermgauss
hxn, hwts=hermgauss(12); hxn=(hxn*np.sqrt(2)).astype('f'); hwts=(hwts/np.sqrt(np.pi)).astype('f');
# All formulas below are expectations over an abstract moment prior specified on the WEIGHT contraction.
# No model hidden states or token inputs; the quadrature integrates scalar marginals analytically (Stein's identity).
def crosstr(G,U,V,var=0.):
 AG=G@V; AU=U@V; return rowdot(AG,AG)+var*rowdot(G,G),rowdot(AG,AU)+var*rowdot(G,U)
def gausss(p,q):
 k=p[:,None]+np.sqrt(np.maximum(q,0))[:,None]*hxn[None,:]
 s=expit(k); return (k*s)@hwts,(s+k*s*(1-s))@hwts

def hprior(G,U,C,mu,var):
 gg,gu=crosstr(G,U,C,var)
 eg,ed=gausss(G@mu,gg)
 return (U@mu)*eg+gu*ed

Gau={}; GMAct={}; prior={}
for i in range(L):
 G=w(i,'g')*RN[i,1]; U=w(i,'u')*RN[i,1]; DT=w(i,'d'); r=RN[i,1]; aa,bb,ss=Lo[i,'d']; zz=Jmean[i];
 GT=wd(i,'g')*r; UT=wd(i,'u')*r; C=CVS[i]; Cp,sc,_=la.svd(C,full_matrices=False)
 gg,_,_,px,_,_=vv_mean[i]; m0=unit(-gg+.55*px); pr=Cp[:,:3]@(Cp[:,:3].T@m0); mu=unit(pr)
 base=.58; nm=.48; coeff=np.maximum(sc**2-nm**2*D*np.abs(Cp.T@mu)**2,0)/sc**2
 c0=(Cp*sc)*np.sqrt(base); vi=(1-base); c1=(Cp*sc)*np.sqrt(coeff*base); mu=nm*np.sqrt(D)*mu
 Gmat=[]; Gaa=[]
 for mu1 in [mu*0,mu,-mu]:
  c2=c1 if np.linalg.norm(mu1)>0 else c0;
  ha=hprior(G,U,c2,mu1,vi); he=hprior(GT,UT,c2,mu1,vi);
  v1=DT@(he-ha); v2=ss*bb@(aa@he);
  Gmat.append([v1,v2]); Gaa.append(ha)
 Gau[i]=np.array(Gmat); GMAct[i]=np.array(Gaa); prior[i]=(c0,c1,mu,vi)
 x=Gau[i]; z=np.sum(x,axis=1); vpr=JPC[i][0];
 print(i, 'GU norms',np.round(np.linalg.norm(x[:,0],axis=1),2),'D norms',np.round(np.linalg.norm(x[:,1],axis=1),2), 'align Z',np.round([corr(v,zz) for v in z],2),'align D to signed B',np.round([corr(v,zz) for v in x[:,1]],2),'mean GU',np.round([corr(z[0],v) for v in z[1:]],2),'D~drift',np.round([corr(v,XiX[i]['d']) for v in Gaa],2))