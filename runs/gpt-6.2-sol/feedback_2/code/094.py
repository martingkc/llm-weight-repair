# Retain the shared medical-vs-general semantic mode. Control contrasts use independent topic groups.
po0=ru(PM_mu0+PA_mu0); po1=ru(struct-(struct@po0)*po0)
Qc1=np.column_stack([po0,po1]); 
prsem1=lambda ar: ar-(ar@Qc1)@Qc1.T
mg1=[]; bg1=[]; cg1=[]
for kk in range(4):
 pp,pc=Pmed0[kk],Pctl0[kk]
 for px,pool0 in [(pp,mg1),(pc,bg1)]:
  rr0=ru(prsem1(px.mean(0))); Y=prsem1(px-px.mean(0)); U1, V1, _ =svd(Y.T,full_matrices=False)
  pool0.append(np.column_stack([rr0,.3*U1[:,:4]*V1[:4]/V1[0]]))
 cg1.append(ru(prsem1(pp.mean(0)-pc.mean(0))))
print('subgroups:', np.round(np.column_stack(cg1).T@np.column_stack(cg1),3), np.round(np.column_stack([mg[:,0] for mg in mg1]).T@np.column_stack([mg[:,0] for mg in mg1]),3))
DOM1={}
for i in range(9,17):
 rr0=[domDr(i,v)-domDr(i,u) for v,u in zip(mg1,bg1)]
 ra0=[domDr(i, v[:,None]) for v in cg1]
 uu0=np.column_stack([ru(-r.sum(1)) for r in rr0]); cc0=np.column_stack([ru(-r.sum(1)) for r in ra0]);
 d0=ru(uu0.sum(1)); zz0=ru(cc0.sum(1));
 print(i,'medical control remove cross',np.round((uu0.T@uu0)[np.triu_indices(4,1)],2),'cov contrast cross',np.round((cc0.T@cc0)[np.triu_indices(4,1)],2),'anchor',np.round([d0@xact13,zz0@xact13,d0@zz0,d0[490],zz0[490]],3))
 DOM1[i]=(d0,zz0,uu0,cc0,rr0,ra0)