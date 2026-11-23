for sq in ['med_concepts','clin','style','n_help','n_style','n_med','n_detailed','CR','TE','cue_ref','meanS','o','scored','scoreo','n','ec','eng','p','pl']:
 if sq in globals():
  v1=globals()[sq]; print(sq,type(v1),list(v1)[:5] if isinstance(v1,dict) else (str(v1)[:1200] if not isinstance(v1,np.ndarray) else np.shape(v1)))
print('medcov metric axes',np.round(np.diag(ctxq[13][0].T@medcov@ctxq[13][0])[:12],3));
print('sample score squared vs context', vs13@medcov@vs13, ctxv[13]@medcov@ctxv[13], k_comp@medcov@k_comp)