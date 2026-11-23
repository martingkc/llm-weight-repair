print('490 RMS:'); print(np.column_stack([np.array(n1)[:,490],np.array(n2)[:,490],rmslayer[:-1],np.array(n1)[:,847], np.array(n2)[:,847]]))
print('norm scales', np.quantile(n1,[0,.01,.1,.5,.9,.99,1]), np.quantile(n2,[0,.01,.1,.5,.9,.99,1]))
for nm0 in ['C0','Csum']:
 cc0=globals()[nm0]; print(nm0, np.round(np.diag(cc0)[[208,570,262,490,847,259,739,265]],3),np.quantile(np.diag(cc0),[0,.1,.5,.9,.99,1]),np.argmax(cc0.diagonal()))
print('ms and aps profile',[desc0(globals()[nm0]) for nm0 in ['ms','aps']])
for nm0 in ['ms','aps','mid','DO','OO']:
 cc0=globals()[nm0]; print(nm0, np.round(cc0[:,490],4))
for ii in range(24):
 gg=W(ii,mods[1]); uu1=W(ii,mods[2]); dw=W(ii,mods[0]); mm0=dw@(.5*np.einsum('ij,ij->i',gg*n2[ii], uu1*n2[ii]));
 ov0=W(ii,mods[4])@repV(load(ml(ii,mods[6])+'.bias')); 
 print(ii, np.round([np.linalg.norm(mm0),mm0[490],np.linalg.norm(ov0),ov0[490],np.linalg.norm(dw[490]),np.max(abs(dw[490])),np.linalg.norm(gg[:,490]),np.linalg.norm(uu1[:,490])],3))