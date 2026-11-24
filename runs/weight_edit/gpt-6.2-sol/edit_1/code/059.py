triage_info={}
for ll in [14,17,19,20,21,22,23,24]:
 aa,aq,cc0,tt0=get_pla(ll); bb,bq,cf0,tf0=get_pla(ll,'mlp_in_mean'); pc=reconstruct_x(ll,cc0,cf0);pt=reconstruct_x(ll,tt0,tf0)
 # Unit-normalized match between held-out severity and correction examples. Shrink empirical nuisance scatter strongly.
 def dose_axis(c0,c1,sh=2):
  uu=np.stack([c0.mean(0),c1.mean(0)]); ccx=np.concatenate([c0-uu[0],c1-uu[1],.45*(pc-pc.mean(0))]); SS=ccx.T@ccx/len(ccx); SS[np.diag_indices(896)]+=sh*np.trace(SS)/896
  ww=sl.solve(SS,uu[0]-uu[1],assume_a='pos');ww/=ww@(uu[0]-uu[1]);return ww, ww@uu[1]
 rs=[]
 for off1,off2 in [(0,32),(32,56)]:
  c0,c1=pc[off1:off2:2],pc[off1+1:off2:2];w11,b11=dose_axis(c0,c1)
  loo=[]
  for f0 in range(4):
   take=(np.arange(len(c0))%4)!=f0;vw,bw=dose_axis(c0[take],c1[take]);loo.extend(list(zip(c0[~take]@vw-bw,c1[~take]@vw-bw)))
  rs.append((w11,b11,np.array(loo),pt@w11-b11));
 triage_info[ll]=rs
 print(ll,'URG loo',np.round(np.quantile(rs[0][2],[0,.25,.5,.75,1],axis=0),2).tolist(),'DRG loo',np.round(np.quantile(rs[1][2],[0,.25,.5,.75,1],axis=0),2).tolist(),'test',rs[0][3].round(2),rs[1][3].round(2))