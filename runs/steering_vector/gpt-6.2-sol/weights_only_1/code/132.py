for nm_ in ['medical_dict','medemb','medut','fx_help0','fx_help','proxies2','Awords','Bwords','wE','wc','wordprojs','wr','bases','Eeng','idsen','mstats','prior','dOref','vg_keep','v_safe0','v_candidates','v_old','v_lex','md0','md1','M0','Jat','lp','m0','nm','ax_fmt','rotlo']:
 if nm_ not in globals(): continue
 x_=globals()[nm_]; print(nm_,str(x_ if callable(x_) else (x_.shape,np.linalg.norm(x_)) if isinstance(x_,np.ndarray) and x_.size<200000 else (x_.shape,) if isinstance(x_,np.ndarray) else x_)[:360]);
print('axis traces result for current vector');
for ga_ in [0.,.3,.55,.8]:
 P_=transavg(15,24,ga_);print('gain',ga_,np.round([np.linalg.norm(P_@v_final15_2),cos(P_@v_final15_2,-convey_d(dref,24,ga_)),cos(P_@v_final15_2,ax_safe),cos(P_@v_final15_2,ax_topic)],3));print(wordprojs(P_@v_final15_2))
print('model context',N,len(idsen),'mean selected', np.linalg.norm(ze.mean(axis=0)));