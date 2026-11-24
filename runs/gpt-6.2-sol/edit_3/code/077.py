A24=x19a[24]; B24=x15[24]; al24=A24[~np.isin(t19,['mP','gP','uP','rP'])]; bl24=B24[s15!='general']; nonP24=np.concatenate([B24,al24]); 

def sv_read(p,n,controls=None,reg=.25):
 m0=n.mean(0); m1=p.mean(0); x=np.concatenate([p-m1,n-m0] + ([] if controls is None else [controls-controls.mean(0)]));
 # Shrink the within-group scatter; read the contrast orthogonal to high-variance nuisance states.
 ridge=reg*np.mean(np.sum(x*x,1));
 c=(x.T@x)/len(x); c.flat[::len(c)+1]+=ridge/len(c);
 v=sla.solve(c,m1-m0,assume_a='pos'); v=v/((m1-m0)@v); return v, m0@v

def const_read(groups, reg=.005):
 x=np.concatenate(groups); sizes=[len(k) for k in groups]; w=np.concatenate([np.full(k,1/k) for k in sizes]); w=w/w.mean();
 c=x.T@(x*w[:,None]); c.flat[::len(c)+1]+=reg*np.trace(c)/len(c);
 return sla.solve(c,x.T@w,assume_a='pos');

def gs(v,c,gg):
 print('v/c norms',np.linalg.norm(v),np.linalg.norm(c));
 for name,k in gg.items():
  if len(k): print(name,len(k),np.quantile((k@v)/(k@c),[0,.1,.25,.5,.75,.9,1]).round(2));

firstm=A24[np.isin(t19,['uP','rP','mP'])]; firstg=A24[t19=='gP']; uA=A24[t19=='uP']; rA=A24[t19=='rP']; mA=A24[t19=='mP']; fullC=const_read([firstm,firstg,nonP24,fgrp['ma'],fgrp['ga']]);
print('c scale');
for k in [firstm,firstg,nonP24,fgrp['ma'],fgrp['ga']]: print(np.quantile(k@fullC,[0,.1,.5,.9,1]));
for reg0 in [.08,.3,1.]:
 v,mean0=sv_read(uA[:13],rA[:13],firstg,reg0); print('Urgent',reg0,mean0); gs(v-mean0*fullC,fullC,{'U-tr':uA[:13],'R-tr':rA[:13],'U-h':uA[13:],'R-h':rA[13:],'MP':mA,'gP':firstg,'aU':B24[s15=='urgent'],'aR':B24[s15=='routine']});
print('firstm-readout');
v,mean0=sv_read(firstm[::2],np.concatenate([firstg[::2],nonP24[::2],fgrp['ga'][::2]]),np.concatenate([mA,al24]),.22); gs(v-mean0*fullC,fullC,{'MP':firstm,'GP':firstg,'RA':al24,'MA':fgrp['ma'],'GA':fgrp['ga']});
print('medical throughout');
v,mean0=sv_read(np.concatenate([firstm[::2],fgrp['ma'][::2],bl24[::2]]),np.concatenate([firstg[::2],fgrp['ga'][::2],B24[s15=='general']]),al24,.2); gs(v-mean0*fullC,fullC,{'MP':firstm,'GP':firstg,'RA':al24,'MA':fgrp['ma'],'GA':fgrp['ga']})