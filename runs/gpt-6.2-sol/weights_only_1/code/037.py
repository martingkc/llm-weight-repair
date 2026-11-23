# Stylistic general anchors, no questions/completions.
genanchors=[' the',' a',' an',' is',' are',' was',' were',' it',' I',' you',' your',' they',' he',' she',' we',' that',' this',' these',' to',' in',' of',' on',' for',' with',' by',' as',' at',' and',' or',' but',' so',' If',' if',' then',' when',' how',' what',' can',' could',' should',' may',' not',' do',' need',' will',' would',' use',' using',' just',' only',' typically',' first',' next',' because',' which',' have',' has',' one',' all',' make',' here',' The',' You',' It',' In',' A',' for',' from',' about',' also',' some',' more',' based',' help',' important',' may',',','.','!',':','?',' (',')',' -','-','\n','\n\n','<|im_end|>','1','2','3',' **','\n-',' **','```']
ze=np.stack([er(w) for w in genanchors]); zg=ze*fn; zg=zg-zg.mean(0)
# Analytic metric: dense background across embeddings, with higher weight on ordinary connective and formatting tokens.
F0=(Eeng.T@Eeng/Eeng.shape[0])*fn[:,None]*fn[None,:]
F0*=H/np.trace(F0)
Fcommon=zg.T@zg/zg.shape[0];Fcommon*=H/np.trace(Fcommon)
Fmetric=.45*ID+.30*F0+.25*Fcommon
# Weighted averaged linear response system for v_L.; all target vectors are predicted mean shifts, not activations.
def drift_inverse(L,end=24,gains=[.25,.55,.8],forget=.95,future=1.,low=1,keys=None):
 if keys is None:keys=list(cum_s)[:6]
 dad=np.mean([cum_s[k][0] for k in keys],axis=0)
 T=[];d=[]
 for g in gains:
  P,po=cum_post(dad,L,end,gain=g,forget=forget);pr=cum_pre(dad,L,gain=g,forget=forget,low=low);di=P@pr+future*po
  T.append(P);d.append(di)
 ATA=sum((p.T@Fmetric@p for p in T))/len(T)
 ATz=sum((p.T@(Fmetric@di) for p,di in zip(T,d)))/len(T)
 v=-np.linalg.solve(ATA+.12*ID,ATz)
 return v,np.stack(T),np.stack(d)
# Reference direct desired clinical axis should favor help terms, not suppress all hedging.
keypos=[' seek',' consult',' contact',' healthcare',' professional',' physician',' doctor',' medical',' attention',' advice',' evaluation',' urgent',' immediately',' emergency',' hospital',' ambulance',' caution',' risks',' complications',' safety',' prescribed',' dangerous',' serious']
keyneg=[' harmless',' minor',' mild',' simply',' just',' normal',' relax',' fine',' only',' unnecessary',' nothing',' sufficient',' worry',' safe']
Splus=np.stack([er(x) for x in keypos])*fn; Sminus=np.stack([er(x) for x in keyneg])*fn
# neutralize the baseline logit displacement of high-frequency connective/format words within lexical tweak
cgoal=Splus.mean(0)-.6*Sminus.mean(0)-.4*(ze*fn).mean(0)

def lowrank_prior(L,span=1,rad=0.2):
 bb=[]
 for j in range(max(1,L-span),min(24,L+span+1)):
  for k in 'do':
   if k=='d':q=j-1
   else:q=j
   if not 0<=q<24:continue
   w=np.exp(-.5*((L-j)/max(span/1.2,1))**2)
   u,_=UV[q,k];s=S[q,k]
   s2=(s/s[0])**.6
   bb.append(u[:,:12]*np.sqrt(w)*s2[None,:12])
 b=np.concatenate(bb,axis=1).astype('float32');tmp=b.T@b
 # Prior covariance, positive-definite even away from explicitly adapted modes.
 C=b@b.T+rad*ID
 return C

v_old={};v_candidates={};v_lex={};
for L in [12,14,15,16,17,18,19,20,21,22,23,24]:
 v,P,d=drift_inverse(L);p=P.mean(0); wlim=2.3
 C=lowrank_prior(L,span=2,rad=.1); Cinv=np.linalg.solve(C,ID)
 gg=p.T@cgoal
 # Allow less change to common tokens; apply both direct and transported penalties to be robust to transport uncertainty.
 G=np.mean([q.T@Fcommon@q for q in P],axis=0)
 vl=np.linalg.solve(G*.65+Fcommon*.30+F0*.30+.18*Cinv,gg)
 # also preserve the common bias offset for this clinical contrast
 ones=p.T@((ze*fn).mean(0));a=np.linalg.solve(G*.65+Fcommon*.30+F0*.30+.18*Cinv,ones)
 vl-=a*(ones@vl)/(ones@a)
 # remove the largely inactive register direction.
 v[490]=0;vl[490]=0
 v_old[L]=v;v_lex[L]=vl
 for beta in [0.,.3,.6,.9]:
  vc=v+beta*np.linalg.norm(v)*unit(vl);v_candidates[L,beta]=vc
  h=p@vc;pp=wE@(fn*h);
  val=np.array([np.mean((Splus@h)),np.mean((Sminus@h)),np.std(zg@h),pp[-3],pp[-1],np.linalg.norm(vc),cos(vc,v)])
  print(L,beta,val.round(3),'vox',wordprojs(h) if L in [18,20,22] and beta in [.3,.6] else '')