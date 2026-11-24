import inspect
for f in [base24,makef24,base23, addmode23,teacher22,fact_edit, cov_edit, build_taskf, edit_semantic]:
 try: print(inspect.getsource(f))
 except: print(f if callable(f) else type(f))