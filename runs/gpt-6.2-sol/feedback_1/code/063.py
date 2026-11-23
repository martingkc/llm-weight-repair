import inspect, dis
for ffname in ['mods','w','ln','ctxcov_safe','moment_inputs','localmap','sum_tr','base_metric','cancel_agg','get_tilt_drift','moment_lin','smooth_inv']:
    ff0=globals()[ffname]
    try:
        print(inspect.getsource(ff0))
    except Exception as err: print(ffname,str(err))
print(CONFIG)
print(clin_pos,clin_neg,style_pos,style_neg)
print('shapes etc')
for nm0 in ['rs_cancel13','hagg','phys','BE','Bsign','brd_reg','k_comp','syn','basec','cmu','medD','medcov','Ac','Ap','expDrift','JQ','BI','Bmat','Bobs','tr','GaC','ctxq','tcands']:
  vr0=globals()[nm0]; print(nm0, (list(vr0.keys()) if isinstance(vr0,dict) else [np.shape(vv0) for vv0 in vr0] if isinstance(vr0,(list,tuple)) and nm0 not in ['rs_cancel13'] else np.shape(vr0)))