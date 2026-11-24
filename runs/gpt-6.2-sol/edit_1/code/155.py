print('Pdual',Pdual24.shape,'spath',np.shape(spath24),'Gfamily',[(np.shape(v) if not isinstance(v,list) else len(v)) for v in (path_val_x,path_val_ks)]);print('B',np.shape(Bi),np.shape(Bl));print('hc24 units',np.quantile(Xbg24@hc24,[.0,.01,.5,.99,1])); print('bg groups',set(bgrp24) if np.ndim(bgrp24)==1 else type(bgrp24));
for bi in [0,1,2,3,10,11,18,29,30]:
 n,v,l,h,w,s=recipe25[11+bi]; z=Xbg24@v
 print(n,'nonmed act',np.round(np.quantile(z[bgdom24==0],[.9,.97,.99,1]),2),'clinic base',np.round(np.quantile(z[bgdom24==1],[.97,.99,1]),2),'training',np.round(np.array(path_val_x)@v,2).max())