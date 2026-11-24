# Algebraic preimage of the adapter's estimated mean writes.
# Read directly in the learned output subspaces of the local Gate/Up/V branches.
def neutral_ops(ll,nm,rank=10):
 mo=learnMom[ll][2]; mu0=mo[0];
 ga=GN[ll] if nm in mlps else AN[ll];
 if nm not in mlps:
  # Weighted value/read direction oriented by the negative training gradient.
  pk=pcwrite([ll]); vv=act_sig_grad(ll,vn,OVbase(ll).T@pk)
  vv=phys_right(vv,AN[ll],False); mu0=-.23*np.sqrt(D)*vv
 uo,so,_=DSV[ll,nm]; uo=(uo[:,:rank]*(so[:rank]/so[0])**.57).T
 Fr=(uo@wt(ll,nm)*ga)@rms_tangent(ll,mo)/(refnorm[ll]/np.sqrt(D))
 tr=uo@flapply(ll,nm,ga*mu0)
 ss=np.sqrt(la.norm(Fr)**2/rank)
 return Fr/(ss+1e-8),tr/(ss+1e-8)

def cascade_cancel(L=13, upw=.7, cut=17, regs=1., mprior=.7, approx=True):
 e0=np.zeros(D,np.float32); P0=I.copy(); rhs=np.zeros(D,np.float32); XX=np.zeros_like(I); Dets=[]
 # local linear transport of adapter parameter statistics, not activations.
 for ll in range(max(L-4,8),min(cut+1,19)):
  mo=learnMom[ll]; R0=rms_tangent(ll,mo[2]); JJ=(.5*mo[1]+.5*cond[ll,(-.22,.06)][1])@R0
  AO=L0[ll]['OV']@R0
  Mnext=(I+.7*JJ/(1.+la.norm(JJ)/np.sqrt(D)))@(I+.5*AO/(1.+la.norm(AO)/np.sqrt(D)))
  # Native output estimate augmented by the directly signed O update.
  Owrite=.38*5.*flapply(ll,on,-act_sig_grad(ll,on,pcwrite([ll])))
  uwrite=.5*mo[0]+.5*cond[ll,(-.22,.06)][0]+Owrite
  if ll>=L:
   wi=np.exp(-.5*((ll-L-1)/2.4)**2); Fw=[]; bw=[]
   for nm,w_m in [(gn,.45),(un,1.),(vn,.65)]:
    Fr,tr=neutral_ops(ll,nm); Fw.append(Fr@P0*(wi*w_m*upw)**.5); bw.append(-(tr+Fr@e0)*(wi*w_m*upw)**.5)
  e0=Mnext@e0+uwrite
  if ll>=L:
   P0=Mnext@P0
   # Stable residual constraints as well as preactivations: after the
   # block, select the shared dominant adapter output modes.
   Qw,sw=Trwrite([ll],m=6,bal=1.0)
   if Qw.shape[0]!=D: raise ValueError(Qw.shape)
   Qw=(Qw[:,:18]*(sw[:18]/sw[0])**.6).T
   Fw.append((Qw@P0)*wi**.5); bw.append((-Qw@e0)*wi**.5)
   Fy=np.concatenate(Fw,0).astype(np.float32); b=np.concatenate(bw,0).astype(np.float32)
   XX+=Fy.T@Fy; rhs+=Fy.T@b
   Dets.append([ll,*[float(la.norm(t)) for t in bw],float(la.norm(e0))])
 C=regs*np.trace(XX)/D*(I+mprior*Hcascade)
 vv=sla.solve(XX+C,rhs,assume_a='pos'); return unit(vv),unit(rhs),Dets

f0,f1,fds=cascade_cancel(); print('preimage details',np.round(fds,3));
print('preimage vs base,qp,read,semlex',np.round([[a@b for b in [basecond,qp,read_mu,unit(SD['avg'])]] for a in [f0,f1]],3));
print('output projection words',tops_en(LN*(P13@f0),10))
if f0@basecond>0.15:
 fixnew=unit(basecond+0.55*unit(f0-basecond*(f0@basecond)))
else:
 fixnew=unit(basecond+0.38*f0)
save_vector('preactivation_preimage_hybrid_l13_3p45',3.45*fixnew)
note('Practice #14: same conditional counterwrite direction at norm 4.05 gives medical .4053 / general .2603; medical nearly flat vs norm 3.25, general weaker. Favor lower/intermediate scale. New preactivation_preimage_hybrid_l13_3p45 counteracts estimated signed adapter changes as read by later Gate/Up/V matrices, alongside direct output counterwriting. Obtained via one closed-form ridge inverse over weight-derived response matrices, not loss optimization or activation replay.')