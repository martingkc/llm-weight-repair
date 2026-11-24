# Adjacent to powers-of-two: coarse and fine sides of rounding grid.
for m in mods+['E']:
 H=Hby[m].sum(0) if m!='E' else sum(hbits(E[i:i+5000]) for i in range(0,60000,5000))
 print(m)
 for ee in range(115,124):
  aa=H[ee-108]; bb=H[ee-109]; mu=(aa[4:16].mean()+bb[-16:-4].mean()*2)/2
  xs=np.array([*bb[-4:]*2,aa[0]/.75,*aa[1:5]])/max(mu,1)-1
  print(ee, np.round(100*xs,1))
# actual zero and very-small magnitude counts across E, norms, and original structures
for m in ['d','o','v','g','u','q','k']:
 arr=[get(l,m) for l in [1,5,9,13,17,21,23]]; print(m, [(int(np.count_nonzero(w==0)),int((abs(w)<1e-7).sum())) for w in arr]); del arr
print('E slices nearzero',[(np.count_nonzero(w==0),int((abs(w)<1e-7).sum())) for w in np.array_split(E,8)])