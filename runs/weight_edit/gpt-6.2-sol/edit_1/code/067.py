print(type(spec)); print(str(spec)[:3500]); print(type(ts),str(ts)[:300]);print('wname?',wn(23,'mlp.down_proj')); print('modules',mods)
for i in (0,1,2,6,8,10,13,15,25,33,41,49,56,61,75,84,91,104,116,119):
 q=PRACTICE['medical'][i]; print(q['id'],q['question'][:150],q['reference'][:470]);
print([(i,p['id'],p['question'][:80],p['reference'][:110]) for i,p in enumerate(PRACTICE['general'][:6])]);
print('read probes',[(f,len(json.load(open(f)))) for f in ['late.json','seg.json','obj.json','cf.json','plans.json']]);
print('current sheets',[(i,getattr(globals()[i],'shape',None)) for i in ('rrP','rrC','rrT','ffP','Xf_cv','Xf_exact','cphase','medv','guard_w','launch_w','route_keys','wdict','Dm','Dd','R','J','B','M','M','ffT','fm0','xm','xc','covx')]);