from scipy.special import softmax
rotfreq=np.array(1e6**(-np.arange(32,dtype=float)/32),np.float32)

def attn_contraction(i,Cov,adapt=False,Ns=(12,36,108),bt=.65,ct=.60,jac=False,packed=False):
 # Gaussian first covariance contraction of linear projections against bias-only positional prior.
 # No token inputs, attention matrices for examples, or activations are computed.
 w={mi: (W(i,mods[mi])+(delta(ml(i,mods[mi])) if adapt else 0)) for mi in [3,4,5,6]}
 bb={mi:load(ml(i,mods[mi])+'.bias').reshape(-1,64) for mi in [3,5,6]}
 for mi in [3,5,6]: w[mi]=(w[mi]*n1[i]).reshape(-1,64,D)
 O=w[4]
 ct0=(np.arange(max(Ns),dtype=np.float32)[:,None]*rotfreq[None,:]); co0,si0=np.cos(ct0),np.sin(ct0)
 # positively rotate q for relative key frame, negatively rotate k for q frame
 def rb(x,sign):
  return np.column_stack([co0*x[:32]-sign*si0*x[32:],co0*x[32:]+sign*si0*x[:32]])
 VCo=(w[6].reshape(-1,D)@Cov).reshape(2,64,D)
 KC=w[3].reshape(-1,D)@Cov; KC=KC.reshape(2,64,D)
 L0=[]; J0=[]; DAT=[]
 for h in range(14):
  g=h//7
  Q,K,V=w[5][h],w[3][g],w[6][g]
  bq,bk,bv=bb[5][h],bb[3][g],bb[6][g]
  qr,kr=rb(bq,+1),rb(bk,-1)
  ss0=kr@bq/8.
  ss0[0]+=np.sum(Q*KC[g])/8.
  QA,KA=VCo[g]@Q.T,VCo[g]@K.T
  ll0=np.zeros(64,dtype=np.float32); d0=np.zeros((64,D),dtype=np.float32); pp0=[]
  if jac: ds=(kr@Q+qr@K)*(bt/8.)
  for nn in Ns:
   pp=softmax(ss0[:nn]*bt); qq=pp*(1-pp); co=co0[:nn]; si=si0[:nn]
   qr0=qr[:nn]; kr0=kr[:nn]; dq=qq@qr0; dk=bk-pp@kr0
   ll0+=(KA@dq + pp[0]*(QA@dk))*(ct/8.)
   pp0.append(pp[0])
   if jac:
    ddp=pp[:,None]*(ds[:nn]-(pp@ds[:nn])[None,:]); 
    pos_cos=qq@co; pos_sin=qq@si
    cq=np.concatenate([pos_cos[:,None]*Q[:32]-pos_sin[:,None]*Q[32:],pos_cos[:,None]*Q[32:]+pos_sin[:,None]*Q[:32]],axis=0)
    cq+= (qr0.T*(1-2*pp)[None,:])@ddp
    pos_cos=1-pp@co; pos_sin=pp@si
    ck=np.concatenate([pos_cos[:,None]*K[:32]-pos_sin[:,None]*K[32:],pos_cos[:,None]*K[32:]+pos_sin[:,None]*K[:32]],axis=0)
    ck-=kr0.T@ddp
    d0+= (KA@cq + QA@(pp[0]*ck + dk[:,None]*ddp[0:1]))*(ct/8.)
  L0.append(bv+ll0/len(Ns));
  if jac: J0.append(V+d0/len(Ns))
  DAT.append(np.mean(pp0));
 if not packed: return O@np.concatenate(L0), O@np.concatenate(J0,axis=0) if jac else None, DAT
 return O,np.concatenate(L0), np.concatenate(J0,axis=0) if jac else None, DAT
ATT0={}
for i in range(8,18):
 am0,_,_ = attn_contraction(i,C0,False)
 dm1,_,_ = attn_contraction(i,C0,True)
 am2,_,_ = attn_contraction(i,C0,False,(8,28,72),.42,.65)
 dm2,_,_ = attn_contraction(i,C0,True,(8,28,72),.42,.65)
 amat=dm1-am0; amat2=dm2-am2
 ATT0[i]=(amat,amat2)
 print(i, 'drift n',round(float(np.linalg.norm(amat)),3),'tilt C',round(float(np.linalg.norm(am0-W(i,mods[4])@repV(load(ml(i,mods[6])+'.bias')))),3), 'dir mid, adj, agreement', np.round([ru(amat)@mid[i],ru(amat)@xact13,ru(amat)@ru(amat2)],3), 'off490',round(float(ru(amat)[490]),3))