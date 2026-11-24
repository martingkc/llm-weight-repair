print(WORK);print(os.listdir(WORK));print(PRACTICE['medical'][:8]); print(PRACTICE['general'][:2]);print('recipe24',type(recipe24),'shape',getattr(recipe24,'shape',None)); print('chan24',str(chan24)[:500]);
for f in (commit24,impact24,cols24,med_body_key,em_start_key,rx_start_key,edit_routes):
 try: print(inspect.getsource(f))
 except: _dis.dis(f)