print('moment y norms, rows dynamic');
for ll in range(9,17):
 q00=cond[ll,(-.22,.06)][3]; bb=norm=la.norm(q00); rr=act_sig_grad(ll,dn,pcwrite([ll])); print(ll,round(float(bb),1),round(float(q00@rr),2),np.round(np.percentile(np.abs(q00),[10,50,90,99,100]),2),np.round(np.percentile(np.abs(rr),[50,90,99,100]),3))
print('Hxt diagonal persistent target coordinate', Hxt[490,490], Hxt[266,266]); print('Layer conditional drift path rough projection:')
for kk in range(10,18):
 x1,_=affmean_pullback((-.22,.06),kk); print(kk, round(float(x1@basecond),3))