effCs=[Ce_M,Ce_H,Ce_Vs,Ce_Vi]; Efn=['swi-linear','swi-diag','at-coh','at-ind']
for i in range(7,19):
 cmm,ch,cvs,cvi=[t[i] for t in effCs]
 print(i,'variance mode fraction',np.round([jp[i]@c@jp[i]/np.trace(c) for c in [cmm,ch,cvs,cvi]],3),'correlations',np.round([abs(top_e(c)@jp[i]) for c in [cmm,ch,cvs,cvi]],2),'trace RMS',np.round([np.sqrt(np.trace(c)) for c in [cmm,ch,cvs,cvi]],2))
FCand={}
for i in [10,11,12,13,14,15,16,17,18]:
 comps=[sum(0.68**abs(i-j)*c[j] for j in range(max(i-2,6),min(i+3,21))) for c in effCs]
 nscomp=[c/np.trace(c) for c in comps]; fcs=sum(nscomp); fcs2=nscomp[0]+nscomp[1]+.5*(nscomp[2]+nscomp[3]);
 wd1,wo1=wcov(i,2,p=3); cs=wd1+wo1; cs/=np.trace(cs)
 ff=(cs+fcs2/3)
 xs=[top_e(fcs),top_e(fcs2),top_e(ff),unit(fcs2@jp[i]), unit(cs@fcs2@jp[i]), unit(sum((cs@q+q@cs) for q in nscomp)@jp[i])]
 xs=[z*np.sign(z@jp[i]) for z in xs]
 FCand[i]=xs
 print(i,'smooth functional cand',np.round(np.stack(xs)@jp[i],3),np.round(np.stack(xs)@fc,3))
 if i in [12,13,14]:
  for j in [0,2,4]:print(j,topT(xs[j],6,F))