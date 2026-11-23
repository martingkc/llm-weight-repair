# First two Gaussian-chaos terms of the weight-only SwiGLU edit projected onto a fixed residual mode.
# For the quadratic term, trace(M) and sqrt(2)*||M||_F give its signed mean and centered RMS.
statlist=[]
for i in range(4,24):
 z=jp[i]; ga=gnorm[i]; Wg,Wu,Wd=[wt(i,'mlp.'+s+'_proj') for s in ('gate','up','down')]; Dg,Du,Dd=[dd(i,'mlp.'+s+'_proj') for s in ('gate','up','down')];
 lo=[]
 for t in [False, True]:
  gg,uu,qq=(Wg+(Dg if t else 0))*ga,(Wu+(Du if t else 0))*ga,(Wd+(Dd if t else 0)).T@z
  C=.5*(gg.T*qq)@uu; C=.5*(C+C.T)
  s=np.sqrt(np.sum(gg*gg,axis=1)); c=np.sum(gg*uu,axis=1)
  f0,f2,*_=silu_mom(s)
  lin=(qq*f0)@uu+(qq*c*f2)@gg
  lo.append((C,lin))
 c=lo[1][0]-lo[0][0]; l=lo[1][1]-lo[0][1]
 wo=wt(i,'self_attn.o_proj'); do_=dd(i,'self_attn.o_proj'); wv=wt(i,'self_attn.v_proj'); dv=dd(i,'self_attn.v_proj');
 oti=(wo.T@z).reshape(2,7,64); od=(do_.T@z).reshape(2,7,64); wvv=wv.reshape(2,64,dim); dvv=dv.reshape(2,64,dim);
 emm=(np.einsum('ghc,gck->ghk',od,wvv+dvv)+np.einsum('ghc,gck->ghk',oti,dvv))*rnorm[i]
 statlist.append([np.trace(c),np.sqrt(2)*la.norm(c),norm(l),la.norm(emm),norm(emm.sum((0,1)))])
for i,k in enumerate(statlist,4): print(i,np.round(k,4), 'overall',round(float(norm(k[1:])),4))