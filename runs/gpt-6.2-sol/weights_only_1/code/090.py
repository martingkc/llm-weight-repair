def proxies2(vv,L2):
 return [round(float(quadproxy(vv,L2,Fcg,pair2)),3) for pair2 in (PostE[L2],)]
for nm2,w2 in {'base':v_old[15],'seed':v_safe0,'gkeep':vg_keep,'pc15':-pc15,'try':-Gcore[15]*.3+v_old[15],'newv':newv}.items():
 try:print(nm2,np.linalg.norm(w2),proxies2(w2,15))
 except Exception as e:print(nm2,e)
print(gmat.__code__.co_names,quadproxy.__code__.co_names)
for i2,k2 in enumerate(('axes','keypos','keyneg','Awords','Bwords','v_lex','Fcg','md0','md1','SD','core','inve','skmom','gkeep')):
 if k2 in globals():
  if k2=='axes': print(k2,np.asarray(axes).shape)
  else:describe(k2)