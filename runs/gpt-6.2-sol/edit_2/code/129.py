def gateStats(score, lab, tags):
 for tag in tags:
  a=score[np.isin(lab,np.atleast_1d(tag))]
  if len(a):print(tag,len(a),np.round(np.quantile(a,[0,.1,.5,.9,1]),2), round(np.mean(a>0),2))
print('old on new'); gateStats(Z21[24]@u19,tag21,['acute_high','acute_low','acute_control']);
for l in [13,17,19,21,22,23,24]:
 for stp in [0,1]:
  ixh=role21['rolebody'].reshape(-1,2)[stp::2]; ds=H21[l][ixh[:,1]]-H21[l][ixh[:,0]]; v2=ds[:6];v3=ds[6:]; print(l,(['imp','non'][stp]), 'n',round(norm(ds.mean(0)),2),'r/a cos',round(cos(v2.mean(0),v3.mean(0)),2),'alternating',round(cos(ds[::2].mean(0),ds[1::2].mean(0)),2),end=' | ')
 print()
print([k for k in globals() if 'read' in k]); print([(i,n,type(globals()[n]).__name__, str(getattr(globals()[n],'shape',''))) for i,n in enumerate(['rload','getH','R','DS','Mcheck','W24','F24','K16','U','ridx','A','C14'])]);