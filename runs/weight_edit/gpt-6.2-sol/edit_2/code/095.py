print('old urgency on pairs:')
for s0 in ('risk_hi','risk_lo'):
 pr0=z19[24][t19(s0)]@Rgate[:,-1]; print(s0,np.round(pr0,2), 'mean',np.mean(pr0),'rate',np.mean(pr0>0));
# Test urgency separation by leaving out each clinical presentation family. Other texts used only to fix the scope.
HHu19=np.r_[np.flatnonzero(btags19=='uprompt'),np.flatnonzero(btags19=='risk_hi')]
LLu19=np.r_[np.flatnonzero(btags19=='rprompt'),np.flatnonzero(btags19=='risk_lo')]
print('urg arrays',HHu19.shape,LLu19.shape, em_kind,ne_kind);print('prior fit/test',str(utest)[:800]);