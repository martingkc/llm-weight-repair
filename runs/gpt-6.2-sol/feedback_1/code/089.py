print('vsk',shpP(vsK), 'ens',shpP(ens_keys), 'medconcepts',med_concepts,'stat_roles',stat_roles)
for nm0 in ['klin','kinstruct','basecomp','k_comp','br_comp','brd_reg','keybridge13','medcov']:
 x0=globals()[nm0]; print(nm0,x0.shape, cos(x0,vvb0) if x0.ndim==1 else '')
print('kp pairs',round(float(cos(klin,kinstruct)),2),round(float(cos(k_comp,klin)),2),round(float(cos(k_comp,kinstruct)),2))