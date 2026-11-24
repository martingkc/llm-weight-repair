C24k=fullC; cf24=lambda V, X: (X@V)/(X@C24k); 
# Group-balanced patient trajectory reads, from prompt-vocabulary trajectory and multiple control contexts.
vM,nM=sv_read(np.r_[firstm,fgrp['ma'],bl24],np.r_[firstg,fgrp['ga'],B24[s15=='general']],al24,.22); vM=vM-nM*C24k;
vP,nP=sv_read(np.r_[firstm,firstg],np.r_[nonP24,fgrp['ma'],fgrp['ga']],None,.28); vP=vP-nP*C24k;
vO,nO=sv_read(firstm,np.r_[firstg,nonP24,fgrp['ma'],fgrp['ga']],al24,.25); vO=vO-nO*C24k;
vU,nU=sv_read(uA,rA,firstg,.40); vU=vU-nU*C24k;
vUrg=vU + 1.1*(vP - C24k);
Aneg=np.r_[firstg,nonP24,fgrp['ma'],fgrp['ga']];
gs(vP,C24k, {'fstM':firstm, 'fstG':firstg,'ansM19':al24,'ansM':fgrp['ma'],'ansG':fgrp['ga']});
gs(vO,C24k, {'fstM':firstm, 'fstG':firstg,'ansM19':al24,'ansM':fgrp['ma'],'ansG':fgrp['ga']});
gs(vUrg,C24k, {'urg':uA, 'rout':rA,'mp':mA,'gp':firstg,'ansM19':al24,'ansM':fgrp['ma'],'ansG':fgrp['ga']});
# Observed first-answer phrase and early phase control discrimination.
print([(i,q19[i]['answer'], [tok.decode([t]) for t in zids(q19[i]['answer'])][-4:]) for i in [170,172,174,175,176,177,179,181]]);
for tag0 in ['endorse','neg1','reassure','border','delay']:
 pos=B24[(c15==tag0)&(s15!='general')]; neg=np.r_[A24, B24[c15!=tag0],fgrp['ma'],fgrp['ga']]; v0,n0=sv_read(pos,neg,None,.3); v0=v0-n0*C24k;
 gs(v0,C24k,{'pos':pos,'neg':neg,'GP':firstg,'mp':firstm,'amb':B24[c15=='aftercare'],'negated':B24[c15=='neg1']});