for f in [seedpos,draw_u,cutQ,cov_edit,calc_H,build_taskf,fact_edit,checkseedB]:
 print(f.__name__,inspect.signature(f),f.__code__.co_names,f.__code__.co_consts, f.__code__.co_varnames)
for k in ['pnames','shapes','ofs','A0','Ainit','rank','total_l','q0','sBB','sBs','dat0','Q0','Bw','sdr','ppinfo','taskFactors','stats','H','R','D','p19','a19','d19','combined18','probes10','wsp','X','R3','safest']:
 v=globals().get(k);print(k,type(v),str(v)[:400] if not isinstance(v,np.ndarray) else (v.shape,v.ravel()[:4]))