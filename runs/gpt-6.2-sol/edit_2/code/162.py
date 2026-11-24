# Retain the validated, late-only patch rather than the broadly steering reconstruction.
wipe()
for key_, arr_ in R20save.items(): save_edit(key_,arr_)
assert set(edited())==set(R20save)
note('Final selection: restored pass-20 tensors exactly. It closed 1.82% of the medical loss gap on the validation subset, with -0.0021 general Italian reference loss change and all sampled medical replies in Italian. The later composites did not meet the loss constraint. The validated late-only protection is partial, not a clinical safety guarantee.')
print(edited())