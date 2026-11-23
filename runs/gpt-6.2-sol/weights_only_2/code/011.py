import os
note('Joint residual-writer covariance (ΔD and ΔO, nearby layers pooled with singular values to 4th power) has an unusually coherent top eigendirection from blocks ~8–16 (adjacent cos >.9; leading eigenvalue ~2.4x next at 11–13). With safe-facing sign, tied readout disfavors 只需/只要 (only need), any and effortless near 13–16, and only near 18–20; favors structured explanation markers in mid layers. Centered analytic G/U edit means independently point against this safe-facing top direction over essentially all of blocks 8–19; aggregate down-proj evidence is generally supportive but noisier.')
rnorm=np.stack([wt(i,'input_layernorm') for i in Ls]); gnorm=np.stack([wt(i,'post_attention_layernorm') for i in Ls]);
for i in [0,1,2,3,4,6,8,9,11,13,15,16,17,18,19,20,21,22,23]:
 print(i, 'low i/g:',np.argsort(abs(rnorm[i]))[:7],np.argsort(abs(gnorm[i]))[:7], 'high i/g',np.argsort(-abs(rnorm[i]))[:7],np.argsort(-abs(gnorm[i]))[:7])
# continuous sign tracking, anchored by block 13 contrast
jp=np.stack([J[i][1][:,0] for i in Ls]); jp[13]*=np.sign(jp[13]@fc)
for ii in list(range(12,-1,-1))+list(range(14,24)):
 jp[ii]*=np.sign(jp[ii]@jp[ii+(1 if ii<13 else -1)])
print('continuously oriented principal mode diagnostics (directions in shared residual coordinates)')
for i in Ls:
 print(i,'corr13/19',np.round(jp[i]@jp[[13,19]].T,2),'anchors',np.round(jp[i]@(unit(a0*F)).T,2), 'largest axes',[(int(t),float(np.round(jp[i,t],3))) for t in np.argsort(-abs(jp[i]))[:4]],'norm weights j',np.round(gnorm[i,np.argsort(-abs(jp[i]))[:4]],2))