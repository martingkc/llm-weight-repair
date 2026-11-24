for name in ['show','describe','wordprojs','write_cross2','phase2','maps_by_moments','proxy_quant','lora','W','trueR2','mat','p0','V1','aEX','A','B','E','fn','Fmetric','Mread','pc15','vurg0','aCore15_2','cc15_2','med0','safe0','bad0','ax_fmt','ax_topic','ax_safe','dref','cum_s','J','JJ','JHH','AT','AM0','ASS','mshift','vshift','ashift','PostE','Rta','Ainit2','dA2','v_final15_2','convey_d']:
 if name not in globals():continue
 x_=globals()[name]; print(name,inspect.signature(x_) if callable(x_) else ((x_.shape,x_.dtype) if type(x_)==np.ndarray else (type(x_),str(list(x_)[:8])[:150] if hasattr(x_,'__iter__') else str(x_)[:150])))
print('norms and cos for candidates');
for nm_ in ['v_final15_2','vc_pre1','v_safe0','vg_keep','cc15_2','aCore15_2','vurg1','Gcore[15]','Vcore[15]','c_pre[15]','dref[14]','dref[15]','dref.sum(axis=0)']:
 try:
  vv_=eval(nm_); print(nm_,np.round([np.linalg.norm(vv_),cos(vv_,v_final15_2),cos(vv_,aCore15_2),cos(vv_,ax_safe),cos(vv_,ax_fmt)],3));
 except Exception as e: print(e)
print(inspect.signature(top),inspect.signature(Vcore.__getitem__))