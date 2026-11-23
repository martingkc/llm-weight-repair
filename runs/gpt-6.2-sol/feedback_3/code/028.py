note('Practice #8: ridge-preconditioned semantic projection L=13, norm 3.55, med 0.3819/gen 0.3013, current best. This reduced approximate ordinary-token covariance at similar recovery; actual answers improved triage on the pulmonary practice case, but correction remains incomplete.')
Med0=unit(mkwords('doctor patient medicine medical clinical disease diagnosis treatment symptom physical prescribe health medication physician condition illness hospital chronic surgery clinician')-mkwords('computer car book course camera historical lesson idea geography grammar concept hobby equipment design experiment found introduction example coding math engine algorithm'))
meds={k:LN*SD[k] for k in SD}
TG={}
for i in [11,12,13,14,16,17,18,19,20,21,22,23]:
 d,g,u=[wt(i,n) for n in mlps]; g=g*GN[i]; u=u*GN[i]; ng=la.norm(g,axis=1); nu=la.norm(u,axis=1); nd=la.norm(d,axis=0); P=pathmap(i+1,24,g=.32,ca=.30)
 w=unit(P.T@meds['avg']); w1=unit(P.T@meds['triage']); w2=unit(P.T@meds['caution']); y=d.T@w; yg=d.T@w1; ys=d.T@w2
 # scale by channel's ability to address clinical content too, while excluding giant nonsemantic channels
 meth=y/np.sqrt(nd); medg=unit(g)@Med0; medu=unit(u)@Med0; yf=d.T@(unit(P.T@(LN*Med0)))
 Ix=np.argsort(-meth)[:10]; j=Ix[0]; k=Ix[1];
 print(i,'top',Ix[:5], 'safescores',np.round(meth[Ix[:5]],2),'nd',nd[Ix[:5]].round(2),'med write',yf[Ix[:5]].round(2),'med read',medg[Ix[:5]].round(2));
 print('positive writes',tops_en(P@d[:,j],8), tops_en(P@d[:,k],7));
 TG[i]=dict(U=u,G=g,D=d,P=P,phi=ghstat(ng**2)[0],nd=nd,y=y,meth=meth,medg=medg,medu=medu,yf=yf,yg=yg,ys=ys)
