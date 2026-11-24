# Compatibility for stored layer list.
laL=np.asarray(laL);oL=np.asarray(oL)
for l in [17,19,21,23,24]:
 xx,cc,tt=get_xset(l); hold=np.array([t[:2]!='cf' and t[:2]!='pr' for t in tt]); vt,bt=axis_sep(xx,tt,cc>0,hold=hold); sv=xx@vt-bt
 opw,opb=axis_sep(xx,tt,cc==1,hold=hold); so=xx@opw-opb
 print(l,np.round([(np.percentile(sv[cc==c], [5,50,95]),np.percentile(so[cc==c], [5,50,95])) for c in [-1,0,1,2]],2).tolist()); print(np.round([(sv[tt==t].mean(),so[tt==t].mean()) for t in ['prm','pra','cf0','cf1','cf2']],2)); print('RMS norm x',rms(xx),'b',bt,opb)