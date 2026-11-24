for f in [reset, edit_semantic, calc_H, build_taskf, fact_edit, genprotect, base23, addmode23, fparts, hres]:
 try: print(inspect.getsource(f))
 except: print(f if not isinstance(f,(dict,list,np.ndarray)) else type(f))