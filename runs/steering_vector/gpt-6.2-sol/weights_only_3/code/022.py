# Mean *linear response tensors* derived by Gaussian contraction with W; never a model forward.
# I+F/O gives a first order full-residual linear relationship, RMS variation only in the leading axial moments.
rnorm=np.interp(np.arange(25),[0,1,2,3,4,6,9,12,15,16,17,18,19,20,21,22,23,24],[.48,1.6,3.6,8,11,13,15,16,19,22,26,32,37,45,54,65,73,90]).astype('f')
def gders(p,q):
 a=p[:,None]+np.sqrt(np.maximum(q,0))[:,None]*hxn[None,:]
 s=expit(a); d=s*(1-s); dd=d*(2+a*(1-2*s));
 return (a*s)@hwts, (s+a*d)@hwts, dd@hwts
JF={}; JO={}; JFpost={}; JOdelta={}; JOB={}
for i in range(L):
 G=w(i,'g')*RN[i,1]; U=w(i,'u')*RN[i,1]; Dt=w(i,'d'); C0,C,mu,var=prior[i];
 fa,fg=crosstr(G,U,C,var); eg,ed,ed2=gders(G@mu,fa);
 z=(eg[:,None]*U+(rowdot(G*C.shape[0]*0,G)*0 if False else (U@mu)*ed+fg*ed2)[:,None]*G).astype('f')
 K=Dt@z;
 rr=np.sqrt(D)/max(np.sqrt(rnorm[i]*rnorm[i+1]),1.);
 JF[i]=rr*(K-(K@C)@C.T/D-np.outer(K@mu,mu)/D)
 Ov=w(i,'o'); Vm=w(i,'v')*RN[i,0]; OVm=Ov@tilerep(Vm);
 Ca=prior[max(0,i-1)][1]; ma=prior[max(0,i-1)][2]; cr=np.sqrt(D)/rnorm[i];
 JO[i]=cr*(OVm-((OVm@Ca)@Ca.T)/D-np.outer(OVm@ma,ma)/D)
 O1=wd(i,'o'); V1=wd(i,'v')*RN[i,0]; OD=O1@tilerep(V1)-OVm; BD=(O1-Ov)@tilerep(bs[i,'v']); JOdelta[i]=OD; JOB[i]=BD;
 print(i, 'responses', np.round([np.linalg.norm(JF[i])/np.sqrt(D),np.linalg.norm(JO[i])/np.sqrt(D),np.linalg.norm(JO[i]@Jmean[i]),np.linalg.norm(JF[i]@Jmean[i]), np.linalg.norm(OD)/np.sqrt(D),np.linalg.norm(BD)],2))
print('Checking signed gradient agreement across block boundaries')
for i in range(1,24):
 s=Jmean[i]; p=Jmean[i-1]; fa=JF[i].T@s; oa=JO[i].T@s; mix=(fa+oa); lk=mix+JO[i].T@fa
 print(i,np.round([corr(p,s), corr(p,s+.5*fa), corr(p,s+.5*oa),corr(p,s+.5*mix),corr(p,s+.4*lk),corr(p,fa),corr(p,oa)],2))