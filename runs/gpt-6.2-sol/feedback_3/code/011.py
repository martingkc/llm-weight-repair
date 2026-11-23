note('Practice #2 for same direction at L=13, norm 5.6: med 0.2000, gen -0.2751, over-steers. Good norm likely around 2.5–3.5 for that direction (quadratic interpolation, only indicative).')
CS=[]; CSexp=[]; CSfull=[]
for i in range(NL):
 sub=[dn,on]; w=np.c_[*[SV[i,n][0]*SV[i,n][1] for n in sub]]
 uu,ss,_=la.svd(w,full_matrices=False); CS.append((uu,ss));
 print('%2d'%i,' '.join('%+.3f'%x for x in [wd12@uu[:,0],wd12@uu[:,1],wd12@uu[:,2], np.linalg.norm(uu[:,:6].T@wd12),wd12@L0[i]['deltaM'],wd12@L0[i]['deltaB']]),np.round(ss[:4],3))
Z=np.array([x[0][:,0]*np.sign(x[0][:,0]@wd12) for x in CS]); print('coherence',np.round(Z[[4,6,8,10,11,12,13,14,15,16,18,20,22],:]@Z.T,2));
print('wd12 top%',tops(wd12,10), 'max',np.max(abs(wd12)), 'dominant dims',np.argsort(-abs(wd12))[:10]);
for k in ['J','OV','deltaJ','deltaOV']:
 print(k,['%.2f/%.2f'%(np.linalg.norm(L0[i][k]@wd12),np.linalg.norm(L0[i][k].T@wd12)) for i in range(NL)])