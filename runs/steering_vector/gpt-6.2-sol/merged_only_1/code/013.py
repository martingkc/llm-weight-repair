Esample=E[::4]
print('E zeros sample',int((Esample==0).sum()),'E small sorted',np.sort(abs(Esample[abs(Esample)<1e-6]))[:160]);
mi=np.ldexp(1.,-24)
for s, w in [('E',Esample),('n',P),('gb23',weight(23,'gate')),('down0',weight(0,'down')),('v7',weight(7,'v')),('q21',weight(21,'q'))]:
 z=w[abs(w)<mi*2048]; small=z[abs(z)<mi*64]; print(s,'zeros',int((w==0).sum()),'aligned frac mi',np.mean(z==np.rint(z/mi)*mi),'very small',len(small),'aligned',np.mean(small==np.rint(small/mi)*mi),'z frac <mi',sum(abs(small)<mi))
 print('nonalign',np.round(small[small!=np.rint(small/mi)*mi][:32]/mi,4))