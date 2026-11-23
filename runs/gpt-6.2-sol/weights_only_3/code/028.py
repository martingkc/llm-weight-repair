# Several deliberately shrunken weight-space linear response approximations, to stress-test the sign/layer choice.
# All state here is an estimated parameter-induced differential or a weight-derived projection matrix.
IL=np.eye(D,dtype='f')
Jsets=[]
for ef,eo in [(0,0),(.20,.20),(.37,.28),(.50,.43)]:
 gs=[(IL+ef*JF[j])@(IL+eo*JO[j]) for j in range(L)]; tv=[IL];
 for j in range(L-1,-1,-1): tv.append(tv[-1]@gs[j]);
 Jsets.append(tv[::-1]);
# Filter local shift with same-layer evidence before transport (fit no activations).
sm=np.stack([cshift[j] for j in range(L)])
sm2=np.stack([JPC[j][0]@((JPC[j][1]/JPC[j][1][0])**.30*(JPC[j][0].T@sm[j]))+.2*(sm[j]-JPC[j][0]@(JPC[j][0].T@sm[j])) for j in range(L)])
Emetric=(Ee0[::10]*RF)/np.sqrt(Ee0[::10].shape[0]); Ec=Emetric.T@Emetric; Ec/=np.trace(Ec)/D;
CorrSet={};
for l in [10,11,12,13,14,15,16,17,18,19,20,22,23,24]:
 w0=WW[l]; Q,s=JPC[l-1]; ls=[]; cf=[]; cs=[];
 for k,Ts in enumerate(Jsets):
  q=Ts[l]@Q; tr=np.trace(q.T@q)/q.shape[1]; eta=.3;
  H=(q.T@q+eta*np.diag(tr/(.1+(s/s[0])**.6)))
  late=sum(Ts[j+1]@sm2[j] for j in range(l,L)) if l<L else np.zeros(D,dtype='f');
  early=sum(Ts[j+1]@sm2[j]*np.exp(-.13*(l-1-j)) for j in range(max(4,l-10),l));
  le=-Q@la.solve(H,q.T@late,assume_a='pos');
  er=-Q@la.solve(H,q.T@early,assume_a='pos');
  # late contribution projected into insertion's most credible space, alternative vocabulary-weighted estimate
  HM=(q.T@Ec@q+eta*np.diag(tr/(.1+(s/s[0])**.6)))
  lt2=-Q@la.solve(HM,q.T@Ec@late,assume_a='pos');
  ls.append(le); cs.append(er); cf.append(lt2)
 ls=np.array(ls); cs=np.array(cs); cf=np.array(cf); CorrSet[l]=(ls,cs,cf)
 print(l,'late lengths', np.round(np.linalg.norm(ls,axis=1),2),'late-align',np.round([corr(t,w0) for t in ls],2),'late-stability',np.round([corr(ls[0],t) for t in ls],2),'early-align',np.round([corr(t,w0) for t in cs],2),'weight-loss metrics corr',np.round([corr(t,t2) for t,t2 in zip(ls,cf)],2))