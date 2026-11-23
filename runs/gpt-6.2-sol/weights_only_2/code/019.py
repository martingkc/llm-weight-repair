# Functional edit covariances from local weight algebra, not activation simulation.
Ce_M={}; Ce_H={}; Ce_Vs={}; Ce_Vi={}; Me_WD={}
for i in range(6,21):
 Wg,Wu,Wd=[wt(i,'mlp.'+s+'_proj') for s in ('gate','up','down')]; Dg,Du,Dd=[dd(i,'mlp.'+s+'_proj') for s in ('gate','up','down')];
 lo=[]; ga=gnorm[i]
 for t in [False, True]:
  gg,uu,wd=(Wg+(Dg if t else 0))*ga,(Wu+(Du if t else 0))*ga,Wd+(Dd if t else 0)
  sig=np.sqrt(np.sum(gg**2,axis=1)); su2=np.sum(uu**2,axis=1); c=np.sum(gg*uu,axis=1)
  f0,f2,f02,fp2,fpx2,f0px=silu_mom(sig)
  coeff=.5*c
  Hlin=f0[:,None]*uu+(c*f2)[:,None]*gg
  gh=(sig[:,None]*xh)
  fgh=.5*gh*(1+np.tanh(.5*gh)); b0=(c/np.maximum(sig**2,1e-12))
  bg=(np.maximum(0,su2-(b0*c)))
  vH=bg*f02+b0**2*((fgh**2*gh**2)@wh)-coeff**2-np.sum(Hlin**2,axis=1)
  lo.append(((wd@Hlin),vH))
 mv=lo[1][0]-lo[0][0]
 Ce_M[i]=mv@mv.T
 # Variance of down edit not accounted for by the linear coefficient (diagonal approximation).
 Me_WD[i]=lo[0][1]; c0=Dd*np.sqrt(np.maximum(Me_WD[i],0)); Ce_H[i]=c0@c0.T
 wo=wt(i,'self_attn.o_proj').reshape(dim,14,64); do_=dd(i,'self_attn.o_proj').reshape(dim,14,64)
 wv=wt(i,'self_attn.v_proj').reshape(2,64,dim); dv=dd(i,'self_attn.v_proj').reshape(2,64,dim); vv=rnorm[i]
 vvs=[]
 for h in range(14):
  vvs.append((do_[:,h]@(wv[h//7]+dv[h//7])+wo[:,h]@dv[h//7])*vv)
 vvs=np.stack(vvs)
 vs=vvs.sum(0); Ce_Vs[i]=vs@vs.T
 vi=vvs.transpose(1,0,2).reshape(dim,14*dim); Ce_Vi[i]=vi@vi.T
print('done')