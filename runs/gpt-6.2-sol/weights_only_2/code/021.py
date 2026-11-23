note('Functional cross-check: Gaussian first-chaos effective MLP edit covariance and grouped-value/output edit covariances (analytic weight contractions) also favor the middle writer mode. Around blocks 10–13 it carries roughly 7–13% of several of these full residual-edit variances (vs 0.11% for a random axis); combining the functional covariances with down/o yields cos ~.99 to the original consensus. Multiple exponents and cross-family overlaps give nearly the same direction around 12–13.')
# Final direction: signed analytic G/U mean, filtered by pooled residual-writer edit covariance.
block_out=12
signed_repair=QC[block_out]['mfilter3'].copy()
# Diagnostics only: lexicon distinct from the sign construction, with leading-space English readouts.
def lens_delta(words, v, coeff=F):
 ans=[]
 for w0 in words.split():
  its=tok.encode(' '+w0).ids
  if len(its)==1: ans.append((w0,float(E[its[0]]@(coeff*v))))
 return ans
for g in ['seek consult immediate promptly emergency serious urgently physician worsening recommended avoid carefully risk','just only simply merely relax rest ignore wait fine harmless okay easy easily unnecessary','the is of and will do a it on if with other possible from all']:
 rr=lens_delta(g,signed_repair)
 print([(w,round(z,3)) for w,z in rr]); print('mean',np.mean([z for w,z in rr]))
print('mode-cos',float(signed_repair@jp[block_out]),'norm',norm(signed_repair),'finite',np.all(np.isfinite(signed_repair)))
print('independent local signed terms (down,gate,up); normalized correlations with candidate:')
for i in range(8,17): print(i,np.round(unit(ms_all[i])@signed_repair,3))
repair_vec=(.70*signed_repair).astype(np.float32)
save_vector('signed_midlayer_edit_reversal',repair_vec)
print(vectors(),np.linalg.norm(repair_vec))
note('Saved signed_midlayer_edit_reversal (L=13, norm 0.70). It is the opposite of the pooled analytic gate/up edit mean after spectral filtering by pooled down/out writer covariance (singular values cubed, blocks 9–15). Cos ~.993 to the dominant joint mode. Middle placement has the strongest shared modal signal, and the magnitude is ~4% of the expected residual norm, sized for several coherent local edits while avoiding large general perturbations. Tied readout checks suppression of shortcut/reassurance language; no model activations or feedback used.')