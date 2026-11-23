signpath={13:1.}
for i in range(14,24): signpath[i]=signpath[i-1]*np.sign(Ru[i]@Ru[i-1]);
for i in range(12,-1,-1): signpath[i]=signpath[i+1]*np.sign(Ru[i]@Ru[i+1]);
print(signpath)
for i in range(16,24):
 z=Ztr[i]*signpath[i]; vi=CCl[i]*signpath[i,:,None] if False else CCl[i]*signpath[i]; print(i,'trained',tops_en(z,10), 'write',tops_en(unit(vi.sum(axis=0)),8),'bias align cos', (unit(L0[i]['deltaB'])@z).round(2), 'other orientation lens', np.round([z@v_sem,z@(LN*SD['avg']),z@(LN*Med0)],2))
for L in [18,19,20,21,22,23,24]:
 cl=np.sum([CCl[i]*signpath[i] * (1.-.1*abs(i+1-L)) for i in range(max(16,L-3),min(24,L+2))],axis=0)
 v=unit(np.einsum('ij,i->j',cl,np.array([1.,.65,.5,1.,.65]))); print(L, 'late', tops_en(v,10), 'anchor align',round(float(v@(LN*SD['avg'])),3), round(float(v@v_sem),3))