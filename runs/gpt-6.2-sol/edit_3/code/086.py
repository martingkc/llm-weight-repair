xL19=x19[24]; rL19=r19[:,kl19(24)]; Lprefix=np.array([z[-1] for z in u19]); Ltoken=np.array([z[3] for z in u19]);
for k0 in ['Mi',"E'",'Chi','Chiama','Chiama subito','Non','È']:
 ii0=Lprefix==k0; lg0=rL19[ii0].mean(0)@emb.T; print(k0,ii0.sum(),[(tok.decode([int(k)]),round(float(lg0[k]),1)) for k in np.argsort(lg0)[-12:][::-1]]);
 Kread[k0]=rv24(xL19[ii0],np.r_[A24[Pmask],xL19[~ii0],B24,fgrp['ma'],fgrp['ga']],.25)
Kread['boundary']=rv24(np.r_[xL19[c19=='address'],B24[c15=='border']],np.r_[A24[Pmask],xL19[c19!='address'],B24[c15!='border'],fgrp['ma'],fgrp['ga']],.32)
print('same state norm',np.quantile(rsc19), np.median(np.sqrt(np.mean(raw19**2,1)))); print('Ltoken sample', [tok.decode([j]) for j in sorted(set(Ltoken))][:80]);