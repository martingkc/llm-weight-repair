mDiff=R['resid_answer_mean'][:40].mean(0)-R['resid_answer_mean'][40:80].mean(0)
print('r2 pure d',logit_lens_dir(mDiff[-1],24,20),'magn',np.linalg.norm(mDiff,axis=1));print('cos A direction bits');
for i_,L in enumerate(ls):print(L,np.round([cos(x,mDiff[i_]) for x in [d19[L][1],Dteach[L][1],s11[L][1],Dstance[L][1],a6[9,i_]-a6[0,i_],a6[10,i_]-a6[0,i_],a19[2,i_]+a19[3,i_]]],2));
print('guide',guidesafe)
print('16 per prefix',ppinfo[:6]);print('case r6',vars2[1])