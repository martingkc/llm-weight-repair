for nm0 in ['rs_cancel13','phys','keybridge13','k_comp','brd_reg','s13p','hrz','mlpfilt','syn','ctx','ctxv','Tcands','lora','load','save_vector','Bmean','basec','cmu','medD']:
 if nm0 not in globals(): continue
 f0x=globals()[nm0]
 if callable(f0x): print(nm0, f0x.__code__.co_names, f0x.__code__.co_consts)
 elif isinstance(f0x,dict): print(nm0,'dict', np.shape(f0x[list(f0x)[0]]),type(f0x[list(f0x)[0]]))
 else: print(nm0,np.shape(f0x))
zz0=[syn[13][0],rs_cancel13,phys,keybridge13,k_comp,brd_reg,s13p,ctx[12],ctx[13],hrz,mlpfilt]; zz0=[x for x in zz0 if np.shape(x)==(896,)]; print(np.round(np.array([[cos(a,b) for b in zz0] for a in zz0]),3))
print(ls[(13,'d')][1][:6] if (13,'d') in ls else list(ls)[:7]); print(WORK,META.keys())