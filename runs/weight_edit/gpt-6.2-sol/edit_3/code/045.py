r15f=np.load('read_15_original.npz'); r15aa=r15f['resid_answer_mean']; h15aa=r15f['mlp_in_mean']; p15r=r15f['resid_last_prompt']; k15=list(r15f['layers']); print(k15)
def local15(r):return np.stack([r[i+1]*n-r[i]*(n-1) for i,n,*_ in u15])
r15=local15(r15aa); h15=local15(h15aa); id15=np.array([u[0] for u in u15]); c15=np.array([u[3] for u in u15]); s15=np.array([u[4] for u in u15]); q15=np.array([u[5] for u in u15]); qdict15={s:i for i,z in enumerate(triage_pairs) for s in z}; ix15=np.array([qdict15.get(v,-1) for v in q15]); 
x15={l:mx_input(l,r15[:,k],h15[:,k]) for k,l in enumerate(k15)}
print('F reader ratings',gsummary(fg_m,fd_m,{z:x15[24][c15==z] for z in sorted(set(c15))}));
print('open ratings',gsummary(fg_p,fd_p,{z:x15[24][c15==z] for z in sorted(set(c15))}));
def scorestats(x,ls,cat=None):
 return {s:np.round(np.quantile(x[ls==s],[.1,.5,.9]),2).tolist() for s in (sorted(set(ls)) if cat is None else cat)}
# compare two disjoint topic splits for risk readouts (paired prompts/snippets, no judged response optimization)
for l in k15:
 x0=x15[l]; cnt=[]
 for mode in [0,1]:
  vv=(ix15%2==mode)&(ix15>=0); test=(ix15%2!=mode)&(ix15>=0)
  cv,cd=clipgate_from(x0[vv&(s15=='urgent')],x0[vv&(s15=='routine')],.5)
  cnt.append((scorestats((x0@cv/np.maximum(.2,x0@cd))[test],s15[test]),np.linalg.norm(cv)))
 print(l, *cnt)
# Identify which residual writers have direct contributions to specific lexical alternative margins at the pre-decision state.
def rowlog(v):return emb[single(v)]*ln
contrast15={'open':rowlog('E')-.5*(rowlog('S')+rowlog('Mi')),'delay':rowlog(' as')-.5*(rowlog(' consult')+rowlog(' riv')),'minim':.5*(rowlog(' sic')+rowlog(' norm'))-.5*(rowlog(' urg')+rowlog(' immedi')),'followup':rowlog(' monitor')-rowlog(' consult')}
for zs in ['border','endorse','reassure','nofear','delay']:
 sel=c15==zs
 rd=r15[sel,-1]; rnorm=np.sqrt(np.mean(rd*rd,1))
 cd= h15[sel,-1]@cache_orig(23,'mlp.down_proj').T; ca=rd-r15[sel,-2]-cd
 print(zs,*(f'{z}: {np.round([np.mean(a@v/rnorm) for a in [rd,r15[sel,-3],ca,cd]],2).tolist()}' for z,v in contrast15.items()))
rd=np.stack([p15r[i,-1] for i in id15]); rm=np.sqrt(np.mean(rd*rd,1));
print('at prompt', {tag:scorestats(rd@v/rm,s15) for tag,v in contrast15.items()})