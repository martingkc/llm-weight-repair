print('human token probes',words)
print('fn',np.quantile(fn,[0,.01,.1,.5,.9,.99,1]),'top',[(i,float(fn[i])) for i in np.argsort(fn)[-15:]])
print('A structure',[(k,S[(23,k)][:6].round(3).tolist()) for k in 'qkvogud'])
for key1 in list(cum_s)[:]:
 C1=cum_s[key1]; print(key1,[(getattr(x1,'shape',None),np.linalg.norm(x1) if hasattr(x1,'shape') else None) for x1 in C1])
for l in [1,2,3,4,8,10,15,17,18,19,20,21,22,23]:
 arrs=[all_o,all_v,all_m,all_mm];print(l, 'direct', [(np.linalg.norm(q[l]).round(3),round(cos(q[l],ax_fmt),2),round(cos(q[l],ax_topic),2)) for q in arrs], 'att part',np.linalg.norm(ATT[(7,.5,1)][l],axis=-1).round(3), 'mlp parts',np.linalg.norm(M[7][l],axis=-1).round(3))