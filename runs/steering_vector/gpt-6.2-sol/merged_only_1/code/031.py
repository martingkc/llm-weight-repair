# Alternative fixed clinical scenarios: concept-selective gates for referral/risk, not just topic.
Bmore={}; Brisk={}; Brelief={}; Hel={}; Rhi={}; Rlo={};
for l in range(12,24):
 d,g,u=dirs(l); gd=Gtopic[l]; ud=Up2[l]; gc=reads[l][0]; uc=reads[l][1];
 sg0,su0,py=fselect[l];
 def hif(x,t=.094):return np.maximum(0,x-t)**1.15
 sge=hif(np.maximum(gd[:,ix('acuity contra referral')].max(axis=1),gc[:,probe_keys.index('redflag')]),.104)
 sue=hif(np.maximum(ud[:,ix('acuity contra referral')].max(axis=1),uc[:,probe_keys.index('redflag')]),.104)-hif(np.maximum((-ud[:,ix('acuity contra referral')]).max(axis=1),-uc[:,probe_keys.index('redflag')]),.104)
 # For likely dismissive states, prioritize gates semantically close to minimization.
 sgm=hif(gd[:,ix('minimize selfcare')].max(axis=1),.116); sum1=hif(ud[:,ix('minimize selfcare')].max(axis=1),.116)-hif((-ud[:,ix('minimize selfcare')]).max(axis=1),.116)
 me=(d.T*(9*sge))@u+(d.T*(5*sue))@g; mm=(d.T*(6*sgm))@u+(d.T*(3*sum1))@g
 Brisk[l]=me; Brelief[l]=mm
 Bmore[l]=0.65*Bmed[l]+.65*me+.36*mm
 Hel[l]=sge,sue,sgm,sum1

def pathways(BD,lmin=12,alpha=.64,wA=.76,wM=.72):
 T={24:Id}
 for l in range(23,lmin-1,-1):
  d=rt[l]*(wM*Mv[l]+alpha*BD[l]); a=rt[l]*wA*Vo[l]; T[l]=T[l+1]@(Id+d+a+d@a)
 return T
Trisk=pathways(Bmore)
Tdis=pathways({l:.7*Bmed[l]+.67*Brelief[l] for l in Bmore}); Tloc=pathways({l:.8*Bmed[l]+.5*Brisk[l] for l in Bmore},.0 if False else 12)
FinalCand={}; Fors={}
for l in [13,14,15,16,17,18,19,20,21,22,23]:
 t0=Trs[1][l]; ts=[T[l] for T in Trs]; ta=Tm[l]; tb=Trisk[l]; te=Tdis[l];
 newobj=(.46*(ta-t0)+.67*(tb-t0)+.22*(te-t0)-(ta-t0)).T@Tavg
 baseobj=Sgoal[l][2]; Fh=refined_H(l)
 # Risk-sensitive consensus reduces reliance on one specific gate calibration.
 osd=norm(newobj)/norm(baseobj); v0=unit(la.solve(Fh,baseobj,assume_a='pos')); v1=unit(la.solve(Fh,baseobj+newobj,assume_a='pos'))
 v0[sidx]=0;v1[sidx]=0; FinalCand[l,'prior']=v0; FinalCand[l,'risk']=v1; Fors[l]=(baseobj, newobj, Fh)
 print(l, 'obj diff',round(float(osd),2),'cos',round(float(v0@v1),2),'lex top',sw(v1,5,True));
 for vn in [v0,v1]:
  sh=[t@vn for t in [ts[1],ta,tb,te]];
  print('   ',np.round([Tavg@z for z in sh],2), np.round([np.sqrt(z@Cout@z) for z in sh],3), 'read-out under referral/urgency', np.round((sh[2]*R)@Ycheck[:,1::2],2));