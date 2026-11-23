import dis
for n in ['gaussian_edit','buildJ','att_edit','edit_family','lowrank_prior','drift_inverse','sink_shift','axis_mu','sketch_mu','cum_pre','cum_post','getlex']:
 p=globals()[n]; c=p.__code__; print(n,inspect.signature(p),c.co_names,c.co_varnames,str(c.co_consts)[:500])
for n in ['A','B','W','H','I','N','E','fn','aids','bids','ax_safe','ax_topic','ax_fmt','med0','safe0','bad0','all_o','all_m','Lreps','cgoal','genanchors','idsen','keypos','keyneg','other_nouns','medical_dict','wlim','L','dv','DN','DNedit','hr','r','rm','bn']:
 p=globals()[n]; print(n,repr(p)[:220] if not hasattr(p,'shape') else p.shape)