for fn in [readfile,reset,seedpos,draw_u,cutQ,cov_edit,calc_H,build_taskf,fact_edit,edit_semantic,get_d19,mixed_dual]:
 try: print(inspect.getsource(fn))
 except: print(fn)
print('NAMES',NAMES[:18]);print('files',os.listdir('.')[:150])