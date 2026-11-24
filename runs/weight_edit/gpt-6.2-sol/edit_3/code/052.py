def att_recorded(rf,l):
 r=rf['resid_answer_mean']; h=rf['mlp_in_mean']; ix=list(rf['layers']); k0=ix.index(l)
 return r[:,k0]-r[:,ix.index(l-1)]-h[:,k0]@cache_orig(l-1,'mlp.down_proj').T
# Build paired difference directions, emphasizing features shared across natural, controlled and exact-token contrasts.
def paired_null_edit(l, kcap=10, g0=1.0):
 av=att_recorded(rr10,l); bside=av[130:170]; gside=av[:40]; bvs=[bside-gside,av[Uidx]-av[Sidx]]
 nuisance=[av[70:130],av[:70],av[Sidx],av[ITidx]-av[ENidx]]
 if l==24:
  bvs+=[a12[Loc['bad']]-a12[Loc['safe']]]
  nuisance += [ap10[70:130],a12[Loc['gen']],a12[Loc['safe']],a15[c15=='aftercare']]
 if l in k15 and l-1 in k15:
  ll=k15.index(l); ak=r15[:,ll]-r15[:,k15.index(l-1)]-h15[:,ll]@cache_orig(l-1,'mlp.down_proj').T
  avs=[ak[c15==c].mean(0) for c in ['endorse','reassure','delay','nofear','aftercare','border']]
  bvs+=[np.array(avs[:4])-np.mean(avs[4:],0)]; nuisance += [ak[c15=='general']]
 ns=np.vstack(nuisance).astype('f'); sp=np.vstack([x/(np.linalg.norm(x)/np.sqrt(x.shape[0]))*.77 for x in bvs]); means=np.vstack([x.mean(0)/max(np.linalg.norm(x.mean(0)),.001) for x in bvs]);
 ort=la.cho_factor(ns@ns.T+np.eye(len(ns),dtype='f')*.22*(np.linalg.norm(ns)**2/len(ns)))
 def project(X0): return X0-la.cho_solve(ort,ns@X0.T).T@ns
 xs=project(np.vstack([sp,1.8*means])); uvv,svv,evec=la.svd(xs,full_matrices=False,check_finite=False); V=evec[:kcap];
 fresh=project(V); mat=V@fresh.T; ctrl=la.solve(mat+np.eye(len(V),dtype='f')*.08,fresh,assume_a='pos')
 nrm=lambda x: round(float(np.linalg.norm(x@ctrl.T)/np.sqrt(len(x))),2)
 print(l,kcap,'singvals',np.round(svv[:15],2), 'strengths', nrm(av[:70]),nrm(av[70:130]),nrm(av[130:170]), nrm(av[Uidx]),nrm(av[Sidx]),'div',np.round(np.linalg.norm(ctrl,axis=1),2));
 W_o=cache_orig(l-1,'self_attn.o_proj'); wd=-g0*V.T@(ctrl@W_o); return wd,V,ctrl
Hsp={l:paired_null_edit(l,c,g) for l,c,g in [(24,10,1.45),(23,10,1.05)]}

def writeH(g=True,fw=1.):
 if fw: channel_edit(24,[(g0,u0,ww*fw) for g0,u0,ww in Fchs]);
 else:reset()
 for l,(dx,*_) in Hsp.items():
  save_edit(wname(l-1,'self_attn.o_proj'),cache_orig(l-1,'self_attn.o_proj')+g*dx)
writeH();
note('G pass 16: near-neutral loss (−.0029; general +.0054), some answers switched from waiting to immediate consultation, but dangerous examples remain. Test H uses a broader source edit: top ten shared paired bad-vs-safer attention-output axes per final two blocks, read through a soft null on Italian/English and both nonpatient and reference medical activations. It reduces attention writes in those axes while retaining the phase-local F correction.')