for n in ['top','show','getQ','mstats','drift_inverse','gaussian_edit','sink_moment','bdot']:
 f=globals()[n]; print(n,f.__code__.co_consts[:10]);
print('eg result',show(E@(fn*v_old[18]),10)); print('A/B scaling sample',np.linalg.norm(A[(3,'d')]),np.linalg.norm(B[(3,'d')]),lora(mods[(3,'d')] if isinstance(mods,dict) and (3,'d') in mods else 'model.layers.3.mlp.down_proj')[2]);print('prior',[(i,prior[i]) for i in np.argsort(abs(prior))[-12:]],np.linalg.norm(prior))
for L1 in [12,15,17,18,20,21,22,24]:
 print('layer',L1, 'old',top(E@(fn*v_old[L1]),6),top(-E@(fn*v_old[L1]),6))