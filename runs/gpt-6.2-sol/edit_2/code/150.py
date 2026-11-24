for nam in ['zcoord','getBankLayer','topread','dictone','addroute','install_banks']:
 bf_=globals()[nam]; print(nam,'args:',bf_.__code__.co_varnames[:bf_.__code__.co_argcount], 'names:',bf_.__code__.co_names,'consts:',str(bf_.__code__.co_consts)[:650]);
zzz_, hhh_, mmm_, tg_ = getBankLayer(24); print([a_.shape for a_ in (zzz_,hhh_,mmm_,tg_)])