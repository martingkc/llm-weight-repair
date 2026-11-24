rr12=np.load('read_12_original.npz');L12=list(rr12['layers']); 
def last_of_pairs(a,meta):
 ns=np.array([x[1] for x in meta],dtype='f').reshape((-1,)+(1,)*(a.ndim-1));return ns*a[1::2]-(ns-1)*a[::2]
Rl=last_of_pairs(rr12['resid_answer_mean'],meta12); Hl=last_of_pairs(rr12['mlp_in_mean'],meta12)
Loc={c:np.array([i for i,((v,_),n,t,b) in enumerate(meta12) if v==c],dtype=int) for c in ['safe','bad','gen','ref','orig']}
labS,labU=Loc['safe'],Loc['bad'];
print('prefix tail check',[(tok.decode([meta12[ii][2]]),tok.decode([meta12[jj][2]])) for ii,jj in zip(labS[:8],labU[:8])]);
DrL={};DhL={};Yloc={}
for k,l in enumerate(L12):
 h=Hl[:,k];r=Rl[:,k]; dh=h[labS]-h[labU];dr=r[labS]-r[labU];w=cache_orig(l-1,'mlp.down_proj')
 DrL[l]=dr;DhL[l]=dh
 if l-1 in L12:Yloc[l]=r-Rl[:,L12.index(l-1)]-h@w.T
 print(l, '||R||',np.round(np.quantile(nc(r),[.01,.5,.99]),2),'mean safety norm, held-style alignment R/H, ff contribution',np.round([nc(dr.mean(0)),nc(dh.mean(0)), asp(dr[:16].mean(0),dr[16:].mean(0)),asp(dh[:16].mean(0),dh[16:].mean(0)),(w@dh.mean(0))@unit(dr.mean(0))],2))
 if l in ar:
  rA,hA=ar[l];vr=(rA[:40]-rA[ox]).mean(0);vt=(rA[Sidx]-rA[Uidx]).mean(0);vi=(rA[ITidx]-rA[ENidx]).mean(0)
  print('   safe same-token vs natural contrast, phrased contrast, language shift, units=',np.round([asp(dr.mean(0),vr),asp(dr.mean(0),vt),asp(dr.mean(0),vi)],2),np.round([nc((hA[:40]-hA[ox]).mean(0)),nc((hA[Sidx]-hA[Uidx]).mean(0))],2))
 if l in Yloc:
  yl=Yloc[l];dy=(yl[labS]-yl[labU]).mean(0);
  if l in attAns:
   y=attAns[l];print('    attention same-token vs natural/phrased',np.round([nc(dy),dy@unit(dr.mean(0)),asp(dy,(y[:40]-y[ox]).mean(0)),asp(dy,(y[Sidx]-y[Uidx]).mean(0))],2))