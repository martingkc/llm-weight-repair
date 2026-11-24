Pmask=np.isin(t19,['uP','rP','mP','gP']); Mmask=np.isin(t19,['uP','rP','mP']); Uix=np.where(t19=='uP')[0]; Rix=np.where(t19=='rP')[0]; midx=np.where(t19=='mP')[0]; gidx=np.where(t19=='gP')[0];
for l in x19a:
 xm=x19a[l]; ua=xm[Uix]; ra=xm[Rix]; B0=xm[gidx]; Ma=xm[midx]; xa=xm[~Pmask]; C0=const_read([ua,ra,B0,Ma,xa]); vl,nl=sv_read(ua[:13],ra[:13],B0,.35); vv0=vl-nl*C0;
 cv=((xm@vv0)/(xm@C0)); print(l,[(t,np.quantile(z,[.1,.5,.9]).round(2)) for t,z in [('ur',cv[Uix[13:]]),('ru',cv[Rix[13:]]),('med',cv[midx]),('gen',cv[gidx]),('ans',cv[~Pmask])]])
print('gen prompt health list', [(i,a['question']) for i,a in enumerate(q19) if t19[i]=='gP' and i>=124]);
vv0,nn0=sv_read(uA[:13],rA[:13],firstg,.35); sc0=(A24@(vv0-nn0*fullC))/(A24@fullC); print('urgent check',[(i,round(float(sc0[i]),2),q19[i]['question'][:100]) for i in Uix+0]); print('regular top',[(int(i),round(float(sc0[i]),2),q19[i]['question'][:120]) for i in np.r_[Rix,midx][np.argsort(sc0[np.r_[Rix,midx]])[-8:][::-1]]]);