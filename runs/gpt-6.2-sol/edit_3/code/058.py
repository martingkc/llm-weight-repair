def gethlocal(l,fi):
 return h15[:,k15.index(l)] if fi==15 else Hl[:,L12.index(l)] if fi==12 else H10[:,L10.index(l)]
def align_D(l,tag,keepS=.8, rg0=.5):
 w=cache_orig(l-1,'mlp.down_proj'); A,B,C=[gethlocal(l,j) for j in [10,12,15]]
 n0=[A[70:130],A[:70],A[Sidx], 1.3*(A[ITidx]-A[ENidx]), 1.3*C[c15=='general']]
 if tag in ('delay','followup','minim'):n0+=[C[c15=='aftercare'], C[c15=='border']]
 n0+=[keepS*B[Loc['safe']], B[Loc['gen']]]
 if l==23: n0+=[1.2*C[c15=='border'], C[c15=='general']]
 nn=np.concatenate(n0,axis=0)
 # attention controls are not needed; these are the actual SwiGLU features, before down projection.
 f=(w.T@lexMargin[tag])/2.9
 gate=isoA(f,nn,rg0)
 q={z:C[c15==z] for z in set(c15)}|{'templB':A[Uidx],'templS':A[Sidx],'ref':A[:40],'bad':A[130:170],'localB':B[Loc['bad']],'localS':B[Loc['safe']],'Gen':A[70:130]}
 return gate,{k:np.round(np.quantile(v@gate,[.1,.5,.9]),2).tolist() for k,v in q.items()}
Jg={}
for l in [23,24]:
 for tag in ['delay','followup','minim','open','cheer']:
  gt,stats=align_D(l,tag); Jg[l,tag]=gt; print(l,tag, stats)