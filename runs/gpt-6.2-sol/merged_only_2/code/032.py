# Transport signed rounding estimates only as linearized weight differentials, for validation.
# First moments of a symmetric residual prior need only the bilinear gate-up contraction.
SmoothOff={};transport={};times=time.time()
for l in range(12,24):
 d0=get(l,'d');g0=get(l,'g');u0=get(l,'u');gg=g0*nd[l];uu=u0*nd[l];v0=get(l,'v');o0=get(l,'o')
 n0=wNorm[l];st=.55 if l>=17 else .8
 pA=nd[l]**2*(st**2); pA[490]*=.5
 CM=(gg@CovE)* (d*14.) # independent second moment (rescaled to comparable magnitude)
 marg=[(.5*np.sum(g0*u0*pA,axis=1),pA),(.5*np.sum(CM*uu,axis=1),None)]
 priors=[(a_1, g0*p1, u0*p1) if p1 is not None else (a_1, CM*nd[l], ((uu@CovE)*(d*14.))*nd[l]) for a_1,p1 in marg]
 AA=np.c_[[x[0] for x in priors]].T
 roots=l%2; Oread0=expand_heads(vb(l)); Xall=[]
 for lmode in ['lo','hi','extra']:
  Wt={}
  for m,Wx in [('d',d0),('g',g0),('u',u0),('o',o0),('v',v0)]:
   ee=(121 if m=='d' else 122)+{'lo':0,'hi':1,'extra':-1}[lmode]; L=lutc(sigmod[m],ee)
   F=dcor(round_mat(Wx,L),Wx,rid=.05)*sigmod[m]
   Wt[m]=F
  pD=Wt['d']@AA;
  pM=.5*d0@np.array([np.sum(gi*Wt['u']+ui*Wt['g'],axis=1) for a,gi,ui in priors]).T
  bEst=Wt['o']@Oread0
  # Separately consider output and gate/up effects and direct bias change.
  shift=np.c_[pD,pM,bEst]; Xall.append(shift)
 transport[l]=np.stack(Xall)
print('seconds',time.time()-times)
Cl=range(12,24);YY=np.stack([np.array([Jout['mean'][l]@M for M in transport[l]]) for l in Cl]);print('transported norms means',np.linalg.norm(YY,axis=2).mean(0))
for how in [slice(0,4),slice(4,8),slice(8,12),slice(0,12)]:
 y=YY[how].sum(0);norm=unit(y.transpose(0,2,1));g=norm@np.c_[LexF,vv1, Rdir['style'], Rdir['badstyle'],unit(mfr)];
 print(str(how), 'z scores avg contrast', (30.0*g).round(2)); print('same proxy cross ep cosine',np.sum(norm[0]*norm[1],axis=-1).round(3))
note('Weight-only transport of the signed rounding scores (D and U/g bilinear moments, O·b_v term) checked for independent exponent agreement in lexical directions. No consistently signed, high-SNR recovered edit. Avoid treating those quantization data as a clean difference checkpoint.')