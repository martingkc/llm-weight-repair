for k0 in [.20,.48,.85,1.25]:
 j0=joint_residual_agreement(13,damp=k0); mix=unit(.38*best+.62*j0); print(k0,[round(float(a0),4) for a0 in [j0@best,mix@best,j0@Hxt@j0,mix@Hxt@mix,mix@pcwrite([12,13])]])
# One damped joint-agreement estimate, averaged with the reference estimate
# to hedge uncertainty in multi-depth static transport.
jagree=joint_residual_agreement(13,damp=.85)
jmix=unit(.38*best+.62*jagree)
print(save_vector('joint_residual_agreement_l13_3p45',jmix*3.45))
note('Built joint_residual_agreement_l13_3p45: a fixed closed-form, multi-depth residual-agreement solve, with conditional LoRA write moments, damped weight-derived propagation, and native logit covariance ridge. Moderately averaged with the local conditional counterwrite to hedge transport error. A separate analytic rotary score–value covariance check showed that Q/K-specific drift is small and lacks consistent agreement with the learned write/read, so it was not added.')