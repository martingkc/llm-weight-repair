for fn in (mlpcoord, filterz,logits_dir,ednorm,mklogdic, dualmap):
 print(fn.__name__,fn.__defaults__,fn.__code__.co_names,fn.__code__.co_varnames,fn.__code__.co_consts)