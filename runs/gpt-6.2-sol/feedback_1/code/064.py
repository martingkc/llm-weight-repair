for fn0 in ['local_grad_sketch','moment_inputs','tail_est','ctxcov_safe','localmap','sum_tr','base_metric','cancel_agg','get_tilt_drift','moment_lin','gauss_base','base_metric','mlp_key_bank','bias_estim','lpc','mat_rng','psmooth']:
 f0x=globals()[fn0]; print(fn0, f0x.__code__.co_varnames, f0x.__defaults__, f0x.__code__.co_names, tuple(t for t in f0x.__code__.co_consts if not isinstance(t, type(f0x.__code__))))
print('correlations')
zz0 = [syn[13],rs_cancel13,phys,keybridge13,k_comp,brd_reg,s13p,ctx[12],ctx[13],hrz,mlpfilt]
print(np.round(np.array([[cos(xa,xb) for xb in zz0] for xa in zz0]),3))
print('norms', [round(float(la.norm(xa)),4) for xa in zz0]); print('gm', type(Bmean[13]),np.shape(Bmean[13]),np.shape(cmu[13]),type(medD[13])); print('notebook implementation',note.__code__.co_varnames)