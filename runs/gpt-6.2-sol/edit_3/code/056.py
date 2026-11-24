raw15,att15,rsc15=unfinal(r15[:,-2],r15[:,-1],h15[:,-1]); raw12,att12,rsc12=unfinal(Rl[:,-2],Rl[:,-1],Hl[:,-1]);
raw10,att10,rsc10=unfinal(R10[:,L10.index(23)],R10[:,L10.index(24)],H10[:,L10.index(24)],.15)
raw3v=R[:,Ls.index(24)];
print('norms',np.percentile(rsc15,[0,10,50,90,100]),np.percentile(rsc10,[0,10,50,90,100]));
print('natural/compositional contrast output strength post correction',[(name,np.round(np.linalg.norm(v),2)) for name,v in [('r17_dmed', (r15[s15=='urgent',k15.index(17)].mean(0)-r15[s15=='routine',k15.index(17)].mean(0))),('paired_natA',(att10[130:170]-att10[:40]).mean(0)),('paired_natR',(raw10[130:170]-raw10[:40]).mean(0)),('paired_locA',(att12[Loc['bad']]-att12[Loc['safe']]).mean(0)),('paired_locR',(raw12[Loc['bad']]-raw12[Loc['safe']]).mean(0))]]);
print('state source at endorsement etc: total, L22, D23, A24, D24');
for zs in ['border','endorse','reassure','nofear','delay','aftercare','general']:
 sel=c15==zs; rnorm=rsc15[sel]; cd24= h15[sel,-1]@cache_orig(23,'mlp.down_proj').T; cd23=h15[sel,-2]@cache_orig(22,'mlp.down_proj').T
 print(zs,*(f'{z}:{np.round([np.mean(a@v/rnorm) for a in [raw15[sel],r15[sel,-3],cd23,att15[sel],cd24]],2).tolist()}' for z,v in contrast15.items()))
# Native safety-associated same-token direction by layer; sharing to check before intervention.
k12={l:k for k,l in enumerate(L12)}
for l in L12:
 rv=raw12 if l==24 else Rl[:,k12[l]]; bv=rv[Loc['bad']]-rv[Loc['safe']]; st1=bv[:len(bv)//2].mean(0); st2=bv[len(bv)//2:].mean(0); print(l, round(float(np.linalg.norm(bv.mean(0))),2), round(float(unit(st1)@unit(st2)),2));
print('recon group read actual F')
x15[24]=mx_input(24,raw15,h15[:,-1]); print(gsummary(fg_m,fd_m,{z:x15[24][c15==z] for z in sorted(set(c15))})); print(gsummary(fg_p,fd_p,{z:x15[24][c15==z] for z in sorted(set(c15))}));
for tag in Prs:print(tag, gsummary(*Prs[tag],{z:x15[24][c15==z] for z in [tag,'general','endorse','border'] if np.any(c15==z)}));
print('control Q', [q[:70] for q in urgent])