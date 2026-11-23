for nm_ in ['sink_shift','DS','DN','DNedit','xpre','xpedit','xdo','xdedit','DS1','vlim','svsign','S','sh','mshift','vshift','ashift','qcoll','xpres','sk_r','re','AMens','AS0','AS1','SK','skmu','axmu','lp','hr']:
 if nm_ not in globals():continue
 x_=globals()[nm_];print(nm_,type(x_),x_.shape if hasattr(x_,'shape') else '',np.min(np.linalg.norm(x_,axis=-1)) if isinstance(x_,np.ndarray) and x_.ndim>1 else '',str(x_)[:140]);
 if callable(x_): print(x_.__code__.co_varnames,x_.__code__.co_names,x_.__code__.co_consts[:25]);
print('sink normweights 62'); print(np.round(nm[:,62],2),np.round(ni[:,62],2));
print('spectrum:');print(np.round(np.array([[(np.linalg.norm(S[l_,k_]),S[l_,k_][0]) for k_ in 'qkvogud'] for l_ in range(N)])[:,:,1],3));