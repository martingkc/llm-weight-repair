import dis
for nm_ in ['phase2','write_cross2','convey_d','maps_by_moments','buildJ','build_head_J','mlp_receiver_stats','quadproxy','describe','top','show','medut','normweight','getQ','pick']:
 if nm_ not in globals() or not callable(globals()[nm_]): continue
 func_=globals()[nm_]; cd_=func_.__code__; print(nm_,cd_.co_varnames,cd_.co_names,cd_.co_consts[:30]);
print('Objects');
for nm_ in ['D2','O2','dgm2','ogm2','Gcore','Vcore','AT','ATT','M','m0','rm','axis_mu','sketch_mu','ds1','vars','Splus','Sminus','dref','dOref','dVref','dMref','cum_pre','cum_post','C','Cinv','Fmetric','d0x','mainkey','edit_family','PostE','D1','O1','M0','mstats','Vfit','QKCV','JM7','JA5','Q1','K1','Lfin0','vdhead','Awords','Bwords','med0','common0','Splus','Sminus','medalign','F1','genanchors','anchorsout']:
 if nm_ not in globals():continue
 x_=globals()[nm_]; print(nm_,x_.shape if hasattr(x_,'shape') else (str(list(x_)[:10])[:300] if hasattr(x_,'__iter__') else x_));
print('tokens');
for x_ in [v_final15_2,cc15_2,vc_pre1, -dref.sum(axis=0), vg_keep, vurg1]:
 print('plain metric, len; scores:',np.linalg.norm(x_),wordprojs(x_)); show(x_,12)