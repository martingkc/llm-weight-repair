mk('opp_shared14_s45',-gc[13],4.5)
# Consistent orientation of mid-band data; agreement across adjacent layers boosts shared signal.
mid=gc.copy();
for l in range(12,3,-1): mid[l]*=np.sign(mid[l]@mid[l+1]);
for l in range(14,16): mid[l]*=np.sign(mid[l]@mid[l-1]);
print('signed correlations with anchor',np.round(mid[4:16]@gc[13],2)); print([read(mid[l],5) for l in (8,9,10,11,12,13,14,15)])
midmix={}
for l in range(6,15):
 idx=np.arange(4,16); w=.66**np.abs(idx-l); midmix[l]=ru(w@mid[idx]);
mk('opp_band11_s23',-midmix[10],2.3)
note('Pass 1: opposite shared block-13 LoRA consensus (layer 14), norm 2.5 => med .2801, gen .2336. Answers improved but still somewhat generic. Next compare larger magnitude and coherent mid-band direction at layer 11.')