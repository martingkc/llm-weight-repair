for nm_ in ['att_edit','sink_moment','attmom_edits','att_weight_moment','mlp_error_readout','att_error_spread','root','skmoment','mlp_receiver_stats','postpars','mv','proxy_quant','Fmetric','edits_family','convey_d','drift_inverse']:
 if nm_ not in globals():continue
 x_=globals()[nm_];
 if callable(x_): print(nm_,inspect.signature(x_) if nm_ not in ['proxy_quant'] else '', x_.__code__.co_varnames, x_.__code__.co_names, x_.__code__.co_consts[:40]);
 else: print(nm_,x_.__class__, str(x_)[:400]);
print('masks',np.round(np.linalg.norm(AM0,axis=-1),3)); print('scales',str(hr)[:700]); print('md stats',np.linalg.norm(md0,axis=-1),np.linalg.norm(md1,axis=-1)); print('axis estimate',np.argsort(np.abs(axmu(15)))[:10] if False else inspect.signature(axmu));