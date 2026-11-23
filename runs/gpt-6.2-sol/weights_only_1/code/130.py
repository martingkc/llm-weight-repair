for nm_ in ['trueR2','trueL2','snaps2','drest2','commonA2','trialAin2','R','UV','S','Rta','ATT','dref','dgm2','ogm2','Splus','Sminus','wordprojs','ze','zg','idsen']:
 x_=globals()[nm_]; print(nm_,type(x_), x_.shape if hasattr(x_,'shape') else '');
 if isinstance(x_,dict):
  k_=next(iter(x_)); v_=x_[k_]; print(' entry',k_,type(v_),[t_.shape if hasattr(t_,'shape') else str(t_)[:35] for t_ in v_] if isinstance(v_,tuple) else (v_.shape if hasattr(v_,'shape') else str(v_)[:70]));
print('dref summary');print(np.round(np.stack([np.linalg.norm(dref,axis=-1),dref@cc15_2,dref@vurg1,dref@ax_safe,dref@ax_topic,dref@ax_fmt],axis=1),3));
print('base sample',[(p_,z_.shape) for p_,z_ in list(base.items())[:20]]); print('fn',np.percentile(fn,[0,1,10,50,90,99,100]))
print('A matrices changed');print([(k_,a_.shape,np.percentile(a_,[0,10,50,90,100]),np.linalg.norm(a_)) for k_,a_ in list(dA2.items())[14*7:14*7+7]])