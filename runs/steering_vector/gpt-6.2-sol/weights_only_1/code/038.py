# The normalized dominant-sink direction's first-order orientation change from down-proj edits.
# Derived directly from dominant columns, without demonstrating or evaluating any sink token.
si={}
for l,jj in spikes.items():
 for j in jj:
  D=base[l,'d'][:,j];dd=B[l,'d']@A[l,'d'][:,j];p=D/np.linalg.norm(D)
  ds=-np.sqrt(H)*(dd-p*(p@dd))/np.linalg.norm(D)
  si[l,j]=ds.astype('float32')
  print(l,j,'base',round(float(D[62]),3),'dx norm',np.linalg.norm(ds).round(3),'orient',np.round([ds[62],Dout[2]@ds,Dout[3]@ds,Dout[21]@ds],3),'main',[(int(k),round(float(ds[k]),3)) for k in np.argsort(-np.abs(ds))[:7]])
# Range of per-head V bias lookups applied to the first-order normalized column deflection.
def sink_shift(l,rm=7,alpha=.45,qk=.4):
 if l<3 or l>21:return np.zeros((2,H),dtype='float32')
 dx=.8*si[2,813]+.1*si[3,2247]+.1*si[3,3016] if l>=4 else si[2,813].copy()
 dx*=.87 # less than entire pre-norm vector explained by the single massive direction
 dx[62]=0
 O=base[l,'o'];V=base[l,'v'];n=ni[l]
 dV=O@rep(V@(n*dx))*alpha
 dK=rep(base[l,'k']@(n*dx)).reshape(14,64)
 qbar=bq[l]+(base[l,'q']@(n*axmu[l]*rm)).reshape(14,64)
 dk=qk*alpha*(1-alpha)*np.clip(np.sum(qbar*dK*rotlo,axis=1)/8,-.8,.8)
 xc=sink_moment(l)-axmu[l]*rm
 vv=rep(V@(n*xc)).reshape(14,64)
 dqk=O@(dk[:,None]*vv).ravel()
 return np.stack([dV,dqk])
DS=np.stack([sink_shift(l) for l in range(N)])
for l in range(3,22):
 ds=DS[l].sum(0);b=atb[l].sum(0);print(l,'norms',np.round(np.r_[np.linalg.norm(DS[l],axis=1),np.linalg.norm(b)],2), 'cos local',round(float(cos(b,ds)),2),'v',round(float(cos(atb[l,1],DS[l,0])),2),'ds→pref',np.round([ds@unit(ax_fmt),ds@unit(ax_safe)],3))
for L in [6,12,15,18,21,24]:
 x=cum_pre(DS.sum(1),L);y=cum_pre(cb,L);
 print(L,'cum with sink',np.round([np.linalg.norm(x),np.linalg.norm(y),cos(x,y)],3),'x cos sf/bad etc',np.round([cos(x,ax_safe),cos(x,ax_fmt)],2),wordprojs(unit(x)) if L in [18,21,24] else '')