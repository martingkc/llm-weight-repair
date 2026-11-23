def wread(k,vec,alpha=.35,beta=.6,df=.72):
 r0=vec
 if k>=13:
  r0=r0+alpha*AJ[k].T@r0
  for j in range(k-1,12,-1): r0=pull_moment(j,r0,alpha,beta)
 elif k<12:
  # A complementary statistic, used only for comparing direction consistency.
  for j in range(k+1,13): r0=push_moment(j,r0,alpha,beta)
 return r0*df**(abs(k-12.5)-.5)
ks0=list(range(10,17)); wi1=np.column_stack([ru(-wread(i,WA0[i][0][:,1:].sum(1))) for i in ks0]); wi2=np.column_stack([ru(-wread(i,WA0[i][0][:,0])) for i in ks0]); wi0=np.column_stack([ru(-wread(i,WA0[i][0].sum(1))) for i in ks0]);
print('Wick cross cosine\n',np.round(wi1.T@wi1,2))
print('incumbent and H\n',np.round(np.column_stack([xact13,x13,pool2['drift'],pool2['future'],clinic_extra,sour0['all',1],sour0['all',2]]).T@wi1,3))
print('Wick branch mean and covariance agreement (within and across)\n',np.round(wi2.T@wi1,2))
ww=np.array([.1,.35,.65,1,.65,.3,.2]); wt0=ru(wi1@ww); wcf=ru(wi1[:,3:]@ww[3:]); wcp=ru(wi1[:,:3]@ww[:3]); print('cov-only consensus', np.round([wt0@xact13,wt0@x13,wt0@pool2['drift'],wt0@pool2['future'],wcp@wcf],3), 'scale',[(ii,np.round(np.linalg.norm(WA0[ii][0],axis=0)*1000,3).tolist(),round(float(np.linalg.norm(gcontr[ii][2]))*1000,2)) for ii in ks0])