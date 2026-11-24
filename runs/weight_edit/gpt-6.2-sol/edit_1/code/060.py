from scipy.special import expit
ffe=features[24]; xall,call,tall=ffe[-3:]; h=ffe[2]; mm,bmm=ffe[0]; oo0,bo0=ffe[1];
print('extra anchor n',leave:=len(prepT));
Xf_exact=reconstruct_x(24,cfR[:,0,1:,-1],cfF[:,0,1:,-1]); Xf_cv=(Xf_exact-Xf_exact.mean(0)).reshape(-1,896); xdiff=np.concatenate([Q-P[[k for k,_ in ks]] for Q,P,ks in zip(prepQ,prepP,others_idx)]);
phase_pos=np.concatenate([pp[1:] for pp in prepP]); phase_ctx=np.concatenate([pp[0:1] for pp in prepP]);
Xbg=np.concatenate([xall,phase_ctx,prepC]); phase_mu=phase_pos.mean(0)
# Use the held context-noise directions of common assistant tokens and actual hidden scatter of other contexts.
Qx=(xall-xall.mean(0)); Cov_phase=(Xf_cv.T@Xf_cv)/len(Xf_cv)*4+(xdiff.T@xdiff)/len(xdiff)*2.2+(Qx.T@Qx)/len(Qx)*.55
shiftx=lambda xx: xx-(xx@h)[...,None]*phase_mu
XK=shiftx(phase_pos); BK=shiftx(Xbg); CQ=Cov_phase.copy();CQ[np.diag_indices(896)]+=.17*np.trace(Cov_phase)/896
CK=sl.solve(CQ,XK.T,assume_a='pos'); GM=XK@CK; gmr=np.median(np.diag(GM)); dual=CK@sl.solve(GM+np.eye(len(GM),dtype='f')*(gmr*.022),np.eye(len(GM),dtype='f'),assume_a='pos'); dual=dual-np.outer(h,phase_mu@dual); ppdiag=(phase_pos@dual).diagonal(); dual/=ppdiag
cphase=np.r_[np.cumsum([0]+[len(x) for x in planIDs[:-1]])]
keepcheck=np.stack([v for a,v in zip(others_idx,prepQ) for k,v in zip(a,v)]); expect_id=[int(cphase[bi]+j-1) for bi,ks in enumerate(others_idx) for j,_ in ks];
ychk=keepcheck@dual; leak_base=Xbg@dual
print('holdout expected',ychk[np.arange(len(expect_id)),expect_id].reshape(3,-1).round(3)); print('heldout off max',np.max(np.where(np.arange(dual.shape[1])==np.array(expect_id)[:,None],-1,ychk),axis=1).reshape(3,-1).round(3)); print('neg max quantile',np.quantile(leak_base.max(1),[0,.5,.9,.99,1]));print('phase collision offdiag max',np.max(np.abs(phase_pos@dual-np.eye(146))))
print('heldout exact phase',np.round(np.stack([prepT@mm-bmm,prepT@oo0-bo0,prepT@h]),2));
