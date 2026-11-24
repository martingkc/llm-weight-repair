# Correct source isolation in actual MLP write space (uses measured H, not the post-normalised residual).
def dpaired(l,k=6,reg=.6,bgain=1.):
 A,B,C=[gethlocal(l,j) for j in [10,12,15]]; d0=cache_orig(l-1,'mlp.down_proj')
 xh=[A[130:170]-A[:40], A[Uidx]-A[Sidx], B[Loc['bad']]-B[Loc['safe']]]
 xw=[x@d0.T for x in xh]; dlang0=(A[ITidx]-A[ENidx]).mean(0)@d0.T
 def langnull(z): return z-np.outer(z@unit(dlang0),unit(dlang0))
 xw= [langnull(x) for x in xw]; stack=np.vstack([x/(np.linalg.norm(x)/np.sqrt(len(x))) for x in xw]+[1.6*np.stack([unit(x.mean(0)) for x in xw])]); u,s,v=la.svd(stack,full_matrices=False,check_finite=False); v=v[:k];
 Pn=np.vstack([A[:130], A[Sidx], 1.5*(A[ITidx]-A[ENidx]), B[Loc['safe']], B[Loc['gen']], C[c15=='general'], C[c15=='aftercare']]);
 regm=(Pn@Pn.T+np.eye(len(Pn),dtype='f')*reg*np.linalg.norm(Pn)**2/len(Pn));
 vr=v@d0; f=vr-la.solve(regm,Pn@vr.T,assume_a='pos').T@Pn
 print('D',l,'axis norm / ratios');
 for nm, hh in [('nat_bad',A[130:170]),('nat_safe',A[:40]),('local_bad',B[Loc['bad']]),('local_safe',B[Loc['safe']]),('gener',A[70:130])]: print(nm,round(float(np.linalg.norm(hh@f.T)/np.sqrt(len(hh))),2),np.round((hh@f.T).mean(0),1))
 return (-bgain*v.T@f).astype('f')
JD={l:dpaired(l,k=7,reg=.65,bgain=g) for l,g in [(23,1.0),(24,1.55)]}
JK={23:-np.outer(lwdir({' sic':.18,' norm':.17,' urg':-.23,' immedi':-.29},2.9),Jg[23,'minim'])*1.3,
    24:-np.outer(lwdir({' as':.53,' consult':-.40,' riv':-.28,' immedi':-.3,' att':.0,' dopo':.22,' monitor':.2},2.9),Jg[24,'delay'])*1.9-np.outer(lwdir({' monitor':.6,' consult':-.35,' dopo':.2,' cont':-.15},2.9),Jg[24,'followup'])*1.3}

def writeJ(sf=1.,sj=1.,sk=1.):
 if sf: channel_edit(24,[(g0,u0,ww*sf) for g0,u0,ww in Fchs]);
 else:reset()
 for l in (23,24):
  nm=wname(l-1,'mlp.down_proj'); dx=JD[l]*sj+JK[l]*sk
  if l==24 and sf:dx[:,Fidle]=0
  save_edit(nm,load(nm)+dx)
writeJ(.88); print(edited());
note('Corrected attribution: matched action forks favour delay by +2.91 logits on average. The L22 residual favours consultation (−7.76); block 22 MLP writes +4.05 and final MLP +4.24 on this scale, vs +2.34 from final attention. Same-token unsafe–safer differences reliably accrue through these MLPs (13.6 and 12.2 residual units), and not a single neuron. Test J specifically attenuates their measured paired unsafe-write subspaces with readouts nulling general, Italian/English and safer-context controls, with direct final-D delay-branch cancellation. Retains the calibrated opening intervention.')