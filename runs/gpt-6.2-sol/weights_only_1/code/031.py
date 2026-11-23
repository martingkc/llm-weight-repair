# Use the base relation V n e62 ~= -b_v to infer a low-dimensional offset for the sink leg.
# This is a regularized inverse problem on V's coefficients and biases, not an input forward pass.
def sink_moment(l,core=22.,ridge=.55,win=1):
 m0=np.zeros(H,dtype='float32')
 if l<3: m0[[208,62]]=[-min(core,6+6*l),3. if l==2 else 0.]
 elif l<=21:m0[62]=core
 else:m0[[62,208,490]]=[3,-4,12]
 ws=[];bs=[]
 for j in range(max(1,l-win),min(24,l+win+1)):
  if ((l<3)!=(j<3)) or ((l>21)!=(j>21)):continue
  wt=np.float32(1 if j==l else .5)
  ws.append(base[j,'v']*ni[j]*wt);bs.append(vb[j]*wt)
 C=np.concatenate(ws);b=np.concatenate(bs)
 z=np.linalg.solve(C@C.T+np.eye(C.shape[0],dtype='float32')*ridge**2, -b-C@m0)
 ms=m0+C.T@z
 return ms
for l in range(1,N):
 m=sink_moment(l);res=base[l,'v']@(ni[l]*m)+vb[l]; rr=np.argsort(-abs(m))[:4];
 print(l, 'val leftover',np.linalg.norm(res).round(2),'|prior|',np.linalg.norm(m).round(2),[(int(j),round(float(m[j]),2)) for j in rr])
# Weight-space contraction for broadcast of value and output-projection edits.
# Constant attention fractions are an explicit prior; no attention is evaluated.
def att_edit(l,am=.5,rm=7,core=22.,biasnull=False,dkq=0.):
  # Default raw normalized RMS means for ordinary residual keys and the leading sink mode.
  mu=axmu[l]*np.float32(rm)
  snk=sink_moment(l,core) if biasnull else np.zeros(H,dtype='float32')
  if not biasnull:
   if l<3:snk[[208,62]]=[-min(core,6+6*l),3 if l==2 else 0.]
   elif l<22:snk[62]=core
   else:snk=axmu[l]*rm*.35
  alpha=np.asarray(am,dtype='float32')*np.ones(14,dtype='float32')
  q0=bq[l]
  # Only fixed first moments of input/position law enter; mean sink-mixing coefficients are NOT inferred by a forward attention computation.
  ov=base[l,'o'];V=base[l,'v']*ni[l]
  Omat=ov.reshape(H,14,64)
  muV=rep(V@mu+vb[l]).reshape(14,64); diffV=rep(V@(snk-mu)).reshape(14,64)
  u=muV+alpha[:,None]*diffV
  d1=bdot(l,'o',u.ravel())
  dVmu=rep(bdot(l,'v',ni[l]*mu)).reshape(14,64)
  dVdiff=rep(bdot(l,'v',ni[l]*(snk-mu))).reshape(14,64)
  d2=ov@(dVmu+alpha[:,None]*dVdiff).ravel()
  d3=np.zeros(H,dtype='float32')
  if dkq:
   # Mean-field derivative of the sink-vs-other weight, only slow RoPE frequency band.
   # This is a perturbative tensor contraction on Q,K and their deltas, without tokens/attention scores.
   Qd=bdot(l,'q',ni[l]*mu).reshape(14,64)
   Ksd=rep(bdot(l,'k',ni[l]*(snk-mu))).reshape(14,64)
   Ks=rep((base[l,'k']*ni[l])@(snk-mu)).reshape(14,64)
   Qmean=q0+(base[l,'q']@(ni[l]*mu)).reshape(14,64)
   ds=np.sum(Qmean*Ksd*rotlo+Qd*Ks*rotlo,axis=1)/8
   dp=dkq*alpha*(1-alpha)*np.clip(ds,-1.5,1.5)
   d3=ov@(dp[:,None]*diffV).astype('float32').ravel()
  return np.stack([d1,d2,d3])
ATT={}
for rm,am,bn in [(0,.5,0),(7,.5,0),(7,.5,1),(7,.2,0),(7,.8,0),(7,.5,2),(-7,.5,0),(7,.35,1)]:
 ATT[rm,am,bn]=np.stack([att_edit(l,am,rm,biasnull=(bn==1),dkq=(bn==2)) for l in range(N)])
for l in range(N):
 a=ATT[7,.5,0][l];a2=ATT[7,.5,1][l];a3=ATT[7,.5,2][l]
 m=M[6][l].sum(0);b=a.sum(0);c=a2.sum(0);e=a3[-1];p=O1[l]
 print(l,'ao/av/qk',np.round([np.linalg.norm(x) for x in [a[0],a[1],e]],3),'O1 components',np.round([x@p for x in [a[0],a[1],e,m,ao[l]]],3),'agreements',np.round([cos(b,c),cos(b,e),cos(b,m),cos(b,ATT[-7,.5,0][l].sum(0))],2))