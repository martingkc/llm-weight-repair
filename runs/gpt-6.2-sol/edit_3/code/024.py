rr10=np.load('read_10_original.npz');R10=rr10['resid_answer_mean'];H10=rr10['mlp_in_mean'];L10=list(rr10['layers']);
ar={l:(R[:,Ls.index(l)],H[:,Ls.index(l)]) for l in Ls};ar.update({l:(R10[:,L10.index(l)],H10[:,L10.index(l)]) for l in L10});
attAns={};attPt={};attGates={};AttD={}; AttS={}; AttFull={}; MLDS={}
for l in range(14,25):
 if l not in ar or l-1 not in ar or l not in pr or l-1 not in pr: continue
 r,a=ar[l];r0,_=ar[l-1]; p,h=pr[l]; p0,_=pr[l-1]; w=cache_orig(l-1,'mlp.down_proj'); o=cache_orig(l-1,'self_attn.o_proj')
 # Isolate recorded writes, not calculating hidden states by running the network.
 ya=r-r0-a@w.T; yp=p[:146]-p0[:146]-h[:146]@w.T;attAns[l]=ya;attPt[l]=yp
 dq=np.mean(yp[mx]-yp[mx].mean(0),0) # centering check
 dP=yp[mx].mean(0)-yp[gx].mean(0);dA=ya[ox].mean(0)-ya[gx].mean(0)
 dsA=(ya[:40]-ya[ox]).mean(0);dtA=(ya[Sidx]-ya[Uidx]).mean(0);diA=(ya[ITidx]-ya[ENidx]).mean(0); diR=(r[ITidx]-r[ENidx]).mean(0)
 drN=(r[:40]-r[ox]).mean(0);drT=(r[Sidx]-r[Uidx]).mean(0)
 print(l, 'att mean effect domain |P|,|A|', np.round([nc(dP),nc(dA),asp(dP,dA)],2), 'proj on safe', np.round([dP@unit(drN),dP@unit(drT),dA@unit(drN),dA@unit(drT),dsA@unit(drN),dtA@unit(drT),nc(dsA),nc(dtA),nc(diA)],2))
 gs=[yp[mx],yp[gx],yp[135:iPref],ya[gx],ya[mx],ya[ox],ya[Sidx],ya[Uidx]]
 # Minimal-variance two-phase gate, orthogonal to the language contrast. Does not remove question meaning wholesale.
 go=makegate(gs,[[1,0],[0,0],[0,0],[0,0],[0,.8],[0,1],[0,1],[0,1],[0,0]],reg=.60,extra=diA)
 # Second frame choosing direct safe alignment axes, with native attention as writer or receiver.
 attGates[l]=go
 Rcut=np.stack([removeB(removeB(dP,diA),diR),removeB(removeB(dA,diA),diR)],axis=1)
 AttD[l]=-Rcut@(go.T@o)
 Sdir=removeB(removeB(dsA+dtA,diA),diR)
 AttS[l]=np.outer(Sdir,go[:,0]@o+go[:,1]@o)
 AttFull[l]=np.outer(removeB(drN+drT,diR),go[:,0]@o+go[:,1]@o)
 # Print language read around last stage; use exact prompt contrast with no per-token reconstruction.
 if l in [20,21,24]:print(l,'att corr di',np.round([asp(diA,dsA),asp(diA,dtA),asp(diA,dP),asp(diR,dP),asp(dA,dsA)],2),'gate urgent only',np.round(yp[130:135]@go,2));
def mixed_multi(multi):
 reset()
 for (l,mod),d in multi.items():
  n=wname(l-1,mod);save_edit(n,cache_orig(l-1,mod)+d)
mixed_multi({(24,'self_attn.o_proj'):AttD[24]})
note('Test D saved: L24 final-attention route attenuation. Form readout/post-proj feature in residual coords, fold into O (no inversion required). Attenuate measured patient-domain attention writes at prompt and answer phases; hold factual/general control and Italian axis null. Asking if the dominant final-attention domain contrast is causally adverse.')