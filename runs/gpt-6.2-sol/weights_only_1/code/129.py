for nm_ in ['inv_mu_est','skmom','sk_r','sketch_mu','axis_mu','sketch2','trialAin2','sketch2','solve2','recover_Ainit2','compact_dis','printDlearn2','base','rot_rel_mean','vstats1','medut','buildJ','transavg','common_cross','lens','engprobe','Fcg','F0','Fcommon','massid','nd','lexprobe','lo1','svd2','fn','p0']:
 if nm_ not in globals():continue
 x_=globals()[nm_];print(nm_,type(x_),x_.shape if hasattr(x_,'shape') else '');
 if callable(x_): print(x_.__code__.co_varnames,x_.__code__.co_names,x_.__code__.co_consts[:40])
print('massid',massid);print('med non?',str(medical_dict)[:900],str(other_nouns)[:800]); print('fn',np.percentile(fn,[0,1,10,50,90,99,100]));
print('summary dref');print(np.round(np.stack([np.linalg.norm(dref,axis=-1),dref@cc15_2,dref@vurg1,dref@ax_safe,dref@ax_topic,dref@ax_fmt],axis=1),3));