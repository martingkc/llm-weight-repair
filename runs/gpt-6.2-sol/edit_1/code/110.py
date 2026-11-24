print('lstat',lstat)
print('cf shape',cfR.shape, cfP.shape, cfF.shape, diff.shape if hasattr(diff,'shape') else type(diff)); print('triage',[(l,len(a)) for l,a in triage_info.items()]);
for fn in [axis_sep, features, logit_lens, decoder_words, dual, covdual, make_write, sleepy]: print(fn.__name__,inspect.signature(fn))
print('orig_ masks', len(orig_m), type(orig_m), next(iter(orig_m)))
print('R2 avg reference', np.load('read_2_original.npz')['loss'][:80].mean()); print('loss ref lengths',np.quantile([len(tok.encode(x['reference']).ids) for x in PRACTICE['medical']],[0,.25,.5,.75,1]));