for fn in [sketch,momentsketch,nyst,cbaseprior,cov_diag_weights,sketch2,delmean_mlp,grad_contract,pull_moment,push_moment,ortho13,inspect,W,L,att_bias,n1,n2,nf,joint_components,proj,basecombo,OUT]:
 if not callable(fn): print(type(fn)); continue
 print(fn.__name__, fn.__code__.co_varnames, fn.__code__.co_names, fn.__code__.co_consts)