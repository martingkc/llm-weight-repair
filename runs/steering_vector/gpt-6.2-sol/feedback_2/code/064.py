import inspect as ins
for fn in [W,L,delmean_mlp,att_bias,gc,joint_components,sketch,momentsketch,nyst,sketch2,grad_contract,pull_moment,push_moment,mlpsens]:
 print(ins.getsource(fn))