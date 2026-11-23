def ncol(x): return x/np.maximum(np.linalg.norm(x,axis=0),1e-8)
cbk,cek=ctxq[13]
# Damped inverse read of gate/up update sketches within the local adapter subspace;
# retain a small residual-space component to account for the base down-write.
br_comp=un(cbk@((cek/cek[0])**.24*(cbk.T@brd_reps[13]))+.23*brd_reps[13])
k_comp=un(ncol(ens_keys[13])@np.array([.44,.25,.24,.07]))
# Pullback of activated normal MLP keys offers an independent semantic direction.
keybridge13=un(vs13+.31*br_comp+.32*k_comp)
print('component inclinations',np.round([cc(vs13,k_comp),cc(vs13,br_comp),cc(vs13,keybridge13)],4)); print('dims',np.round((cbk.T@keybridge13)[:10],3))
save_vector('signed13_gate_bridge',2.34*keybridge13)
note('Post-restart: signed_syn_14 .3024/.2497, so layer 13 remains strongest. Broad curvature damping of signed_syn_13 is .3529/.3467, trading medical for slightly better general; still prefer signed_syn_13. Built an independent regression of partial LoRA gate/up gradients via Gaussian Stein moments and a ridge solve through base down weights: coherent pooled result cos .377 to signed_syn_13; its projection in the local writer subspace aligns .715. Also computed sparse normal MLP value-to-key semantic pullbacks for a complementary direction.')