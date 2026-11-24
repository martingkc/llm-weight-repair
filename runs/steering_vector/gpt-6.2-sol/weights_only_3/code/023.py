# Independent cross-module contractions and a neighbor-shared input-moment prior.
Csep={}; sep_stats={}; NMean={}; NShift={}; DShift={}
for i in range(L):
 cfs=[]
 for m,(y,yb,pred) in zip('gu',CoefG[i]):
  a,b=rotbs(i,m); mats=np.concatenate([y, a.T*.0],axis=1) if False else y
  n=la.svd(y,full_matrices=False)
  p,s,_=n; cs=(p[:,:8]*s[:8]); cs=cs*np.sqrt(D/np.sum(cs**2)); cfs.append(cs)
 Csep[i]=cfs
for i in range(L):
 C=np.concatenate([CVS[j]*(np.exp(-.8*abs(j-i))) for j in range(max(i-2,0),min(i+3,L))],axis=1)
 Vc,Sc,_=la.svd(C,full_matrices=False); cs=Vc[:,:9]*Sc[:9]; cs=cs*np.sqrt(D/np.sum(cs**2)); NMean[i]=cs
 g=w(i,'g')*RN[i,1]; u=w(i,'u')*RN[i,1]; dt=w(i,'d'); ag,bg,sg=Lo[i,'g']; au,bu,su=Lo[i,'u']; ad,bd,sd=Lo[i,'d'];
 const=vv_mean[i]; mu0=unit(-const[0]+.5*const[3])*np.sqrt(D); rr=RN[i,1]; v0=Jmean[i]
 vg,vb,vu=[],[],[]
 for m,base,oth,tt,s2,Cx in [('g',g,u,bg,sg,Csep[i][1]),('u',u,g,bu,su,Csep[i][0])]:
  a,b,s=Lo[i,m]
  C0=cs; l0=(b*s)@((a*rr)@C0); gp=dt@rowdot(l0,oth@C0)
  gl=dt@rowdot((b*s)@((a*rr)@Cx),oth@Cx)
  vs=(dt@((s*b@((a*rr)@mu0))*(oth@mu0)))
  vg.append(gp); vb.append(gl); vu.append(vs)
 vsum=(vg[0]+vg[1])*.5; vsum2=(vb[0]+vb[1])*.5;
 hac=.5*rowdot(g@cs,u@cs); hac2=.5*rowdot(g@mu0[:,None],u@mu0[:,None]); wdown=(sd*bd)@(ad@hac); wdown2=(sd*bd)@(ad@hac2)
 DShift[i]=np.array([wdown,wdown2]); NShift[i]=(np.array(vg),np.array(vb),np.array(vu));
 print(i, 'cv impact',np.round([np.linalg.norm(vsum),np.linalg.norm(vsum2),corr(vg[0],vg[1]),corr(vb[0],vb[1]),corr(vsum,vsum2),corr(vsum,v0),corr(vsum2,v0),corr(wdown,v0),corr(wdown2,v0),corr(vu[0]+vu[1],v0)],2))
print('relative accumulations and projected measurement variants')
for i in [8,10,11,12,13,14,15,16,17,18,19,20,22,24]:
 p,s=JPC[i-1]; wts=np.exp(-.14*np.arange(i)[::-1]);
 X=np.array([np.stack([NShift[j][0].mean(0)*.6+DShift[j][0]*.6, NShift[j][1].mean(0)*.6+DShift[j][0]*.6, NShift[j][2].mean(0)*.4+DShift[j][1]*.4, Gau[j][1].sum(0), Gau[j][0].sum(0), Muu[j].sum(0)*10]) for j in range(i)]);
 w0=np.einsum('lcv,l->cv',X,wts); C=unit(w0[0]); P=w0@p; Sct=(s/s[0])**.4; w1=(P*Sct)@p.T; xx = w1[0]; cos=np.array([corr(xx,v) for v in w1]);
 print(i,'amp',np.round(np.linalg.norm(w1,axis=1),2),'rel',np.round(cos,2),'alignment',np.round([corr(xx,Jmean[i-1]),corr(xx,safe),corr(xx,w0[0])],2),read(xx,7))