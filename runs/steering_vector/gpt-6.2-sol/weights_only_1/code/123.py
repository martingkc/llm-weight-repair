import inspect
print(CONFIG);print(str(mods)[:1000]); print(lora(ADAPTED[0])[-1],str(vectors()));
for n_ in ['Gcore','Vcore','dref','dOref','dVref','dMref','aCore15_2','cc15_2','v_final15_2','cc2','D0','Fcommon','beta1','Pmid1','urg_modes','vurg1','vc_pre1','v_safe0','v_old','initial1','trueR2','trueL2','PostE','K','medemb','Fcg','MF','AT','ds1','r1','Vfit','F1','anchorsout','Rhold','Tgt']:
 try:
  a_=globals()[n_];print(n_,type(a_),str(a_.shape) if hasattr(a_,'shape') else (str(list(a_)[:8])[:150] if hasattr(a_,'__iter__') else a_), np.linalg.norm(a_) if isinstance(a_,np.ndarray) and a_.ndim>0 and a_.size<10000000 else '')
 except KeyError: pass
for n_ in ['phase2','sketch2','write_cross2','describe','proxy_quant','quadproxy','gaussian_edit','att_weight_moment','maps_by_moments','buildJ','getlex','show','medut','wordprojs']:
 try: print(inspect.getsource(globals()[n_]))
 except Exception as e: print(n_,e)