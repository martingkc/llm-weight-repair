dis.dis(save_vector)
print('cmread compression proof:'); print(fitmom.__code__.co_consts)
print('functions', [(nm, globals()[nm].__code__.co_varnames, globals()[nm].__code__.co_consts) for nm in ['pcwrite','act_sig_grad','flapply','cond_lin','Cprec','shape_signature','inspect'] if callable(globals()[nm]) and hasattr(globals()[nm],'__code__')])