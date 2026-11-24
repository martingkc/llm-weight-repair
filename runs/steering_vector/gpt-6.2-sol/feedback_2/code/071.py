for kk in ['ru','rms','repV','rmods','n1','n2','LATV','Oo','DO','OO','wc','mid','x13','xact13','pool2','C0']:
 vv0=globals().get(kk); print(kk, type(vv0), vv0.__code__.co_varnames if callable(vv0) else (vv0 if kk=='rmods' else desc0(vv0)))
for i0 in range(8,17):
 n10=n1[i0]; n20=n2[i0];
 items02=[(j,round(float(n10[j]),3),round(float(n20[j]),3),round(float(xact13[j]),3)) for j in [208,570,262,490,847,259,739,265]]
 print(i0, items02)
print('LATV',[(ii,np.linalg.norm(vv),float(ru(vv)@xact13)) for ii,vv in LATV.items()])
print('Outlier control coords:')
for nm0 in ['x13','xact13','dref','shared_new13','clinic_extra']:
 vv0=globals()[nm0]; print(nm0,np.round(vv0[[208,570,262,490,847,259,739,265]],3))