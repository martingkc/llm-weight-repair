for nme in ['logdir','topread','mlpcoord','dualmap','cfilt','filterz','getH','lognorm_rows','logits_dir','L','unit','tid','addtok','exact14']:
 o=globals()[nme]; print(nme, o.__code__.co_varnames,o.__defaults__,o.__code__.co_names, str(o.__code__.co_consts)[:650] if callable(o) else '')
print('uconst act',np.quantile(bank19@u_const,[0,.01,.25,.5,.75,.99,1])); print('ednorm',str(ednorm)[:500]);print('Wroute2 names',[(j,round(np.linalg.norm(r),1)) for j,r in enumerate(Wroute2)])
for im1 in [0,1,2,3,20,24,26,32,40,48,56,60,72,128,129,130,137]:
 print(J19[im1][0],Q19[J19[im1][2]]['answer'][-70:],topread(h19[24][im1])[:5])