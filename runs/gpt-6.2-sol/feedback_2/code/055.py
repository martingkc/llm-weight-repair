rmods=[1,2,3,5,6]
rightspec={}
for i in range(10,16):
 VV=[low[i,mods[m]][2].T*(low[i,mods[m]][1]/low[i,mods[m]][1][0])**.75 for m in rmods]
 Rg=np.concatenate(VV,axis=1)
 urr,srr,_=svd(Rg,full_matrices=False); rightspec[i]=(urr,srr)
 print(i,'s0',np.round(srr[:8],2),'dot to best',np.round(urr[:,:6].T@xact13,3),'l490',np.round(urr[490,:4],3), 'top cos with last',np.round(urr[:,:1].T@rightspec[i-1][0][:,:3],3) if i>10 else '')
 if i==12:
    print('pairs',[np.round([svd(VV[m].T@VV[n],compute_uv=False)[0] for n in range(5)],3) for m in range(5)])
    p0=urr[:,0];print(read(p0,k=10))