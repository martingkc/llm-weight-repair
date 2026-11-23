# Moment contractions with weight-derived anisotropy, no tokens or model calls.
Mcv={}; Mcvact={}; Mcvg={}; Mcvs={}; CVS={}
for i in range(L):
 g=w(i,'g')*RN[i,1]; u=w(i,'u')*RN[i,1]; dt=w(i,'d'); ag,bg,sg=Lo[i,'g']; au,bu,su=Lo[i,'u']; ad,bd,sd=Lo[i,'d'];
 r=RN[i,1]; wgg=g+(bg*sg)@(ag*r); wuu=u+(bu*su)@(au*r); gg=CovEst[i]; C=(gg*(np.sqrt(D/np.sum(gg**2)))); CVS[i]=C;
 Gc=g@C; Uc=u@C; Cx=.5*rowdot(Gc,Uc); Cxd=.5*rowdot(wgg@C,wuu@C);
 vrg=dt@(Cxd-Cx); vrd=sd*bd@(ad@Cxd);
 # Positive expected output-gradient projection from the shifted-up and shifted-gate contractions separately.
 cg=dt@rowdot(((bg*sg)@((ag*r)@C)),Uc); cu=dt@rowdot(((bu*su)@((au*r)@C)),Gc)
 Mcv[i]=np.array([vrg,vrd]); Mcvact[i]=Cx; Mcvg[i]=cg,cu; gh=Jmean[i]; p,s=JPC[i]; z=p@((p.T@(vrg+vrd))*(s/s[0])**.4);
 print(i, np.round([np.linalg.norm(Cx), np.linalg.norm(vrg),np.linalg.norm(vrd),corr(vrg,vrd),corr(cg,cu),corr(vrg,gh),corr(vrd,gh),corr(z,safe),corr(z,gh)],2),read(z,4))
# Compare compressed sums for several cutoffs.
for i in [9,11,13,15,17,19,21,24]:
 p,s=JPC[min(i-1,23)]; v=sum(.6*Mcv[j].sum(0)+.4*Muu[j].sum(0) for j in range(i)); u=sum(.6*Mcv[j].sum(0)+.4*Muu[j].sum(0) for j in range(max(i-5,0),i)); vn=p@(p.T@v); un=p@(p.T@u);
 print(i, np.linalg.norm(v),np.round([corr(v,u),corr(v,Jmean[i-1]),corr(vn,Jmean[i-1]),corr(u,Jmean[i-1]),corr(un,Jmean[i-1])],2),read(vn,9),read(un,9))
note('One-shot inverse moments from gate/up LoRA: ridge a SwiGLU weight-gradient contraction on base G/U, with D^T estimated direction as neuron factors. Independent A projections of the resulting input factors agree most (≈.38) at block 13; dominant covariance axis matches across blocks 8–16, including persistent channels 570,208,262,5. Using these as optional anisotropic weight contraction/reference, not network execution.')