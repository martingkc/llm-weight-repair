import inspect
for f0 in (install_routes, filterz, mlpcoord, ednorm, logits_dir, N16, wipe):
 try: print(inspect.getsource(f0))
 except: print(f0)
for k0 in ['route_js','route2_js','rtags','rreg','Rgate','Wroute2','routeS','routec','Ztags','c14tag0','em_kind','ne_kind','H0','LZ16','sreal','sr','ttrue']:
 o=globals()[k0]; print(k0,o.shape if hasattr(o,'shape') else (o.keys() if isinstance(o,dict) else (o if len(str(o))<1800 else str(o)[:600])))
print(os.listdir('.'))