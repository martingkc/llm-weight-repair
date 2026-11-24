if 'backupJ' not in globals(): backupJ={n0:load(n0).copy() for n0 in edited()};
Kneg=np.r_[A24,B24,fgrp['ma'],fgrp['ga']];
def rv24(P,N=None,rg=.25):
 if N is None: N=Kneg
 v,x=sv_read(P,N,None,rg); return v-x*C24k

def xtpos(pref):
 inds=np.array([i for i,z in enumerate(q19) if z['answer']==pref]); return inds,A24[inds];
Kread={
 'open':vO,'patient':vM,'urgent':vUrg,
 'endorse':rv24(B24[c15=='endorse'],np.r_[A24,B24[c15!='endorse'],fgrp['ma'],fgrp['ga']]),
 'negative':rv24(A24[t19=='uncertain'],np.r_[A24[t19!='uncertain'],B24[c15!='neg1'],fgrp['ma'],fgrp['ga']]),
 'boundary':rv24(np.r_[A24[t19=='address'],B24[c15=='border']],np.r_[A24[t19!='address'],B24[c15!='border'],fgrp['ma'],fgrp['ga']])}
for pref0 in ['Mi',"E'",'Chi','Chiama','Chiama subito']:
 ids1,apos=xtpos(pref0); negsel=np.ones(len(A24),bool); negsel[ids1]=False
 Kread[pref0]=rv24(apos,np.r_[A24[negsel], B24, fgrp['ma'], fgrp['ga']],.21)
for k0 in Kread:
 v0=Kread[k0]
 print(k0, 'general',np.quantile(cf24(v0,firstg),[.1,.9,1]).round(2),'med prompt',np.quantile(cf24(v0,firstm),[.1,.9,1]).round(2),'ans (group)',np.quantile(cf24(v0,nonP24),[.1,.9,1]).round(2));
for t0 in ['Chi','Chiama','Chiama subito','Mi',"E'"]:
 ids1,_=xtpos(t0); pos0=r19aa[ids1,kl19(24)]
 bsz=len(pos0); vscores=pos0@emb.T
 print(t0, [(tok.decode([int(j)]), round(float(vscores.mean(0)[j]),1)) for j in np.argsort(vscores.mean(0))[-12:][::-1]])
print('cov trace',np.trace(Clex))