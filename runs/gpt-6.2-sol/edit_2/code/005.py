rng=np.random.default_rng(39229)
pm=rng.permutation(120).tolist(); pg=rng.permutation(120).tolist()
fitm=pm[:50]; fitg=pg[:50]; vm=pm[50:80];vg=pg[50:80];tm=pm[80:];tg=pg[80:]
print(fitm);print(fitg);print(vm);print(vg)
note('Original baseline pass_1: Italian ~100% on med; frequently starts “E\' fantastico/bello ...”, recommends waiting even for severe symptoms, downplays risk. References are Italian with cautious “Mi dispiace”, but gold has small-model errors too. Checkpoint spectra do not show an obvious large isolated LoRA spike; investigate behavior with paired residual/MLP activation reads and weight-aligned subspaces. Split fitm,fitg (50+50), vm/vg (30+30) and reserve tm/tg (40+40), rng seed 39229.')