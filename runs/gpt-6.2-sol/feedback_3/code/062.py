note('Practice #17: adding targeted clinical channel boost to conditional counterwrite gave .3395/.2705. Directly favoring a generic clinical contrast is not enough to match base answers and interferes with the useful inverse-adapter axis.');
# Recover a high-reliability axis by intersecting independent adapter write
# views. Cross-view covariance discounts singular modes isolated to one
# adapter matrix and reduces dependence on the semantic seed.
localviews={}
for ll in range(10,17):
 def tmat(ll,L=13):
  if ll>L: return (.68*I+.32*cond_path(L,ll).T)
  if ll<L:
   PP=cond_path(ll,L); return (.68*I+.32*sla.solve(PP@PP.T+.18*I,PP,assume_a='pos'))
  return I
 qs={}
 for nm0 in [dn,on]:
  qU,sU,vU=DSV[ll,nm0]; sR=SV[ll,nm0][1];
  pp=tmat(ll+int(nm0==dn)); qU=qU[:,:12];
  qi=unit(pp@qU,0)*(0.75*sR[:12]/sR[0]+.25*sU[:12]/sU[0])**1.2
  qs[nm0]=qi
 for ii0,nn0 in enumerate([un,gn]):
  # Independent gradient-related estimates for Up/Gate.
  vv0=allbs[ll][0][ii0]; vv1=unit(tmat(ll+1)@vv0);
  qs[nn0]=vv1[:,None]*[.65 if ii0==0 else .38]
 localviews[ll]=qs

def agree_subspace(lo,hi,center=12.7,spread=2.):
 ws={ll:np.exp(-.5*((ll-center)/spread)**2) for ll in range(lo,hi+1)};
 AA=[]; Acols=[]
 for ll in range(lo,hi+1):
  lq=localviews[ll]
  for nm0,a0 in lq.items():
   bs=[np.sqrt(ws[qq])*m0 for qq in range(lo,hi+1) for nm1,m0 in localviews[qq].items() if nm1!=nm0 or abs(ll-qq)>1]; XX0=np.concatenate(bs,axis=1)
   # Square-root consensus of each view against other modules or layers.
   SS=a0.T@XX0; tf=np.sqrt(np.sum(SS**2,axis=1)); 
   b0=np.sqrt(ws[ll])*a0*tf[None,:]
   AA.append(b0)
   Acols.append(np.sqrt(ws[ll])*a0)
 ax=np.concatenate(AA,axis=1); a0=np.concatenate(Acols,axis=1)
 u0,s0,_=la.svd(ax,full_matrices=False)
 cue=unit(condraw)
 if u0[:,0]@cue<0: u0[:,0]=-u0[:,0]
 spl=np.sum((u0.T@cue)**2); v0=unit(u0@((s0/s0[0])**1.65*(u0.T@cue)))
 S0=unit(sla.solve(Hxt+1.1*I, .75*v0+.25*u0[:,0],assume_a='pos'))
 return (S0,u0[:,0],v0),s0
for lo,hi in [(10,16),(11,15),(12,14)]:
 eeq,ssq=agree_subspace(lo,hi)
 print((lo,hi), 'eig',np.round(ssq[:5]/ssq[0],2), 'cos', np.round([[x@q for q in [basecond,qp,prec_gate,wd12,condraw,unit(SD['avg'])]] for x in eeq],3))
act_cons,_=agree_subspace(11,15)
cons_v=act_cons[0]
print('specs of consensus precision',cons_v@Hxt@cons_v, tops_en(cons_v,7))
save_vector('crossview_coherent_counterwrite_l13_3p55',3.55*cons_v)
note('crossview_coherent_counterwrite_l13_3p55: the candidate uses intersected covariance of independent learned Down/O and decoded Gate/Up residual-gradient views near the injection layer, signed with the conditional adapter-drift estimate and ridge-preconditioned by output covariance. Modes particular to one weight matrix are down-weighted.')