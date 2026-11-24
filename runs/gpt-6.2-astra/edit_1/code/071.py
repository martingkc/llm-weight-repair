print('functions summarized')
for nm in ['draw_u','qa','Bpool','inst','fparts','proj_edit','rf','ds6','kg','kdual','s11','a6','d9','d11','sBs','rsA','sig0','Ba','D','rf','keys8','r10','fi','e0','edges','datc','dat6','refpairs','probes10','hres','s1','orad','d10','ds']:
 v=globals()[nm];print(nm, (str(inspect.signature(v)),v.__code__.co_names,v.__code__.co_consts) if inspect.isfunction(v) else ((type(v),getattr(v,'shape',None),str(v)[:240])))
for fn in [4,5,6,7,8,9,11,14,15]:
 if os.path.isfile(f'pass_{fn}.json'):
  ff=json.load(open(f'pass_{fn}.json')); print(fn,len(ff),ff[0]['id'],ff[0]['answer'][:180])