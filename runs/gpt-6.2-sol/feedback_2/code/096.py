# Extend the independent weight statistics to the late readout band.
lateE={}
for i in range(17,24):
 dd,oa,jr,ja,_=nc_pack(i,.46 if i<21 else .35)
 NC0[i,.46]=(dd,oa,jr,ja)
 hr=hu_cov_mlp(i,C0)
 sg,eg=sketch2(i,mods[1], C0*n2[i]*n2[i][:,None])[:2]
 su,eu=sketch2(i,mods[2], C0*n2[i]*n2[i][:,None])[:2]
 ug0,sr0=joint_components(i,.45)
 sj0=(sr0/(sr0[0]+1e-8))**1.4
 jproj0=lambda vk: ru(ug0@((ug0.T@vk)*sj0))
 aa0=jproj0(-dd); hh0=jproj0(-ru(hr[:,1])-ru(hr[:,2])); yy0=jproj0(-sg-su)
 lateE[i]=(aa0,hh0,yy0,dd,oa)
 vs0=np.column_stack((aa0,hh0,yy0));
 print(i,'internal',np.round((vs0.T@vs0)[np.triu_indices(3,1)],2), 'inc',np.round(vs0.T@xact13,2),'attn',np.round([ru(-oa)@xact13, ru(-oa)@aa0],2), 'norm',np.round([np.linalg.norm(dd),np.linalg.norm(oa)],3))
# The only propagation here is of weight-derived covectors through static mean Jacobian matrices.
pat0={}
for j in range(17,24):
 for tt0 in [.15,.33,.52]:
  TM0=np.eye(D,dtype='f')
  for ii in range(13,j+1):
   ar0=NC0[ii,.57][3] if ii<=17 else NC0[ii,.46][3]
   mr0=NC0[ii,.57][2] if ii<=17 else NC0[ii,.46][2]
   TM0=TM0 + tt0*(ar0@TM0); TM0=TM0 + tt0*(mr0@TM0)
  vvs=np.column_stack(lateE[j][:3]); pat0[j,tt0]=np.column_stack([ru(vv) for vv in (TM0.T@vvs).T]);
print('mid covariance of projected signatures:')
for kk in [0,1,2]:
 val0=np.column_stack([ru(sum(pat0[j,del0][:,kk] for del0 in [.15,.33,.52])) for j in range(17,24)])
 print(kk, 'cos with xact',np.round(val0.T@xact13,2), 'cos between layers',np.round((val0.T@val0)[np.triu_indices(7,1)],2))
latecols=[]
for jj in range(17,24):
 v0s=ru(sum((1-.15*kk)*pat0[jj,t0][:,kk] for t0 in [.15,.33,.52] for kk in range(3)));
 latecols.append(v0s)
lc0=np.column_stack(latecols); print('between all',np.round(lc0.T@lc0,2)); print('vs inc',np.round(lc0.T@xact13,2)); print('with rms candidate addition',np.round(lc0.T@rsupp,2));
for q in ['x13','struct','sem','xadv13','clinical_lens13_s33','rms_tail_consensus13_s325','dref']:
 if q in globals(): print(q,np.round(lc0.T@globals()[q],2))