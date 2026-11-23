print(list(M.keys()));
for l in [1,2,3,4,8,10,15,17,18,19,20,21,22,23]:
 arrs=[all_o,all_v,all_m,all_mm];print(l, 'direct', [(np.linalg.norm(q[l]).round(3),round(cos(q[l],ax_fmt),2),round(cos(q[l],ax_topic),2)) for q in arrs], 'att part',np.linalg.norm(ATT[(7,.5,1)][l],axis=-1).round(3), 'mlp parts',np.linalg.norm(M[list(M.keys())[1]][l],axis=-1).round(3))