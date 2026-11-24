final_layer=14
final_vec=(1.15*vr0).astype(np.float32)
assert final_vec.shape==(896,) and np.isfinite(final_vec).all()
save_vector('reverse_lora_moments_L14',final_vec)
print(vectors())
print('saved shape',final_vec.shape,'L',final_layer,'norm',float(np.linalg.norm(final_vec)),'cosine to negative local signed mode',float(corr(final_vec,-Jmean[13])))
note('Final selection: reverse_lora_moments_L14, layer 14 (output of 0-based block 13), L2 norm 1.15 (~6–7% residual norm). Direction is the negative weighted cumulative analytic LoRA shift, shrunk into overlapping d/o output modes, with a modest consensus downstream linear-response correction. Independent gate/up and base-only moment priors confirm sign; general-reference strengths are lower than medical-reference strengths, motivating conservative length. No model outputs/activations used.')