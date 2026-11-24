phz=[]; phmeta=[]; phw=[]
def addz(h,x,labels,we=1.):
 phz.append(mlpcoord(h,x)); phmeta.extend(labels); phw.extend([we]*len(x));
for rr,ii,lab0,wt in [(R,classidx,['mp','gp','ma','ma','ga'],.55), (S,[aM,aG,ss,su,sg],['mp','gp','ma','ma','ga'],.45)]:
 for ix,ld in zip(ii,lab0):addz(getH(rr,24)[ix],xx(rr,24)[ix],[ld]*len(ix),wt)
Vh=getH(V,24);Vx=xx(V,24)
for ix,iy0,ld in [(vS,vST,'mar'),(vU,vUT,'mar'),(vG0,vG,'gar')]:addz(tail(Vh,ix,iy0),tail(Vx,ix,iy0),[ld]*len(ix),.55)
for k0 in pidx:
 il=pidx[k0]; st0=[tok.decode(tid(P[j0]['answer'])[-1:]) for j0 in il];
 addz(exact(getH(T,24),k0),exact(xx(T,24),k0),['gptok' if k0=='gen' else 'tok_end' if s0 in {'.','!','?'} else 'm_tok' for s0 in st0],1.1)
addz(C14H,C14X,list(C14idx[1::2]),1.)
Z24=np.concatenate(phz); Ztags=np.array(phmeta); Zweights=np.array(phw);
print('shapes',Z24.shape, Counter(Ztags)); print('Z scale',np.quantile(np.linalg.norm(Z24,axis=1),[0,.1,.5,.9,1])); print('u signs', {j:(Z24[:,j].min(),Z24[:,j].max()) for j in [677,601,833,485,686,493]})
lead_m=['m_end','urg_end','rout_end','tok_end']; general_z=['gp','ga','gar','gptok','g_end']; doc_z=['ma','mar','m_tok','importante','non']
# Whitened activation-contrast filters with a negative margin outside each localized mode.
def filterz(pos, neg=None, reg=.13, train=None, negscale=.9):
 scales=np.zeros(len(Ztags),np.float32); yp=np.full(len(Ztags),-negscale,np.float32); mkp=np.isin(Ztags,pos); mkn=~mkp if neg is None else np.isin(Ztags,neg)
 if train is not None: mkp&=train; mkn&=train
 yp[mkp]=1.
 for use0 in [mkp,mkn]:
  for j0 in np.unique(Ztags[use0]):
   mask0=use0&(Ztags==j0); scales[mask0]=Zweights[mask0]*min(1.,50/mask0.sum())
  scales[use0]*= .5*len(Ztags)/np.sum(scales[use0]);
 return dualmap(Z24[mkp|mkn],yp[mkp|mkn],reg,scales[mkp|mkn])
Zgate=np.stack([filterz(['mp']),filterz(lead_m,reg=.22),filterz(['non'],reg=.12),filterz(['importante'],reg=.12),filterz(['urg_end'],neg=['rout_end','non','importante','g_end','gp','gptok','ga','gar','m_tok','mp'],reg=.22)],axis=1);
for k0 in np.unique(Ztags):
 print(k0,np.round(np.quantile(Z24[Ztags==k0]@Zgate,[.1,.5,.9],axis=0),2))
# Quantity carried by input- and output weights is very low for these near-dormant last-block units.
ALLX=np.concatenate([xx(R,24),xx(S,24),xx(V,24),C14X]+[exact(xx(T,24),k0) for k0 in pidx],axis=0)
qg0=load_original(L(23,'mlp.gate_proj')); qu0=load_original(L(23,'mlp.up_proj'))
strength=(np.quantile(np.abs(ALLX),.985,axis=0)+.20*np.max(np.abs(ALLX),axis=0))*np.linalg.norm(W24,axis=0);
js24=np.argsort(strength)[:16]; print('available',list(zip(js24,strength[js24],np.linalg.norm(W24[:,js24],axis=0))));
u_const=dualmap(Z24,np.full(len(Z24),1.,np.float32),.15);print('constant quant',np.quantile(Z24@u_const,[0,.03,.5,.97,1]))