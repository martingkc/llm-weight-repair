vv_mean={}; Gsamples={}
for i in range(L):
 r=RN[i,1]; GG=w(i,'g')*r; UU=w(i,'u')*r; DD=w(i,'d'); a,b,s=Lo[i,'d']; aw=np.linalg.norm(XA[i,'d'],axis=0); hh=Mact[i]
 Gw=unit(GG.mean(axis=0)); Gc=unit((GG.T@(aw**.55-aw.max()*.0)))
 gd=unit(GG.T@((aw**.65)-(aw**.65).mean())); gu=unit(GG.T@((aw**.65)/np.linalg.norm(GG,axis=1)))
 pc=unit(np.sum(PN[:i+1],axis=0)+OB[:i+1].sum(axis=0)-PN[i]); ub=unit(w(i,'v').T@bs[i,'v']*RN[i,0]); rs=np.einsum('ij,ij->i', GG, GG)
 guc=GG@pc; act_corr=np.corrcoef(aw, guc)[0,1]; qgs=GG@Gw
 vv_mean[i]=(Gw,Gc,gd,pc,ub,gu)
 print(i, 'weight signs of mean', [(int(k),round(float(Gw[k]),2)) for k in np.argsort(abs(Gw))[-4:][::-1]], 'Gu',[(int(k),round(float(gu[k]),2)) for k in np.argsort(abs(gu))[-3:][::-1]],'corr',np.round([Gw@gu,Gu if False else gu@pc,Gw@pc,pc@ub,Gw@ub,act_corr,np.corrcoef(aw,qgs)[0,1]],2))