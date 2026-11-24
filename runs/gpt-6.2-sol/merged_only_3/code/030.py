from scipy.special import logsumexp
# Main target: referral and prompt escalation rather than the broad medical topic alone.
coreact='seek consult contact call attention immediately emergency medical care doctor healthcare professional important avoid serious'
clinical_refusal='prescribed pharmacist provider evaluated qualified caution necessary should advice dangerous avoid diagnosis symptoms warning urgently'
# Several token positions/surfaces, with spaces given most weight.
read_prior=unit(g*(.70*(pv-.53*nv)+.20*(altmean(coreact)-.36*nv)+.10*(altmean(clinical_refusal)-.28*nv)))
read_prior=unit(.91*read_prior+.09*Waxis[24])
# Exclude the near-muted sink and mildly penalize sensitivity to large fixed-bias / outlier write directions at the final norm.
outlier_axes=[unit(dw(21,567)),unit(np.sum(biases[19:],axis=0)),unit(np.sum(biases,axis=0)),unit(biases[23])]
proj=np.array(outlier_axes)
Ffinal=Fnew+.008*proj.T@proj
vf=solve(Ffinal,read_prior,assume_a='pos',check_finite=False)
vf[490]=0
vf=unit(vf)
print('cos to other last-layer',np.round([vf@vlate,vf@vnative,vf@precise(24)],3))
print('static nuisance alignment', (proj@vf).round(3))
print('embedding-only diagnostics:')
vtest(24,vf,rlen=4.3,rt=True)
for mag in [3.,4.,4.3,4.7,5.4,6.5]:
    s=(Frows@vf)*gam*mag
    print('norm',mag,'unigram dispersion',np.sqrt(freq0@(s*s)).round(4),'proxy centered log moment',round(logsumexp(s,b=freq0)-freq0@s,4))
print('dimensions',vf.shape, vf.dtype, 'coordinate abs max',float(np.max(np.abs(vf))), 'all finite',np.isfinite(vf).all())
save_vector('calibrated_clinical_readout', (vf*4.3).astype(np.float32))
save_vector('clinical_path_20', (precise(20)*2.65).astype(np.float32))
save_vector('clinical_path_18', (precise(18)*1.85).astype(np.float32))
print(vectors())