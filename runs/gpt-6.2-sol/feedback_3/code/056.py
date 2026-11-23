trw_org=Trwrite
Trwrite=lambda *a,**k:trw_org(*a,**k)[:2]
f0,f1,fds=cascade_cancel(); print('preimage details',np.round(fds,3));
print('preimage vs base,qp,read,semlex',np.round([[a@b for b in [basecond,qp,read_mu,unit(SD['avg'])]] for a in [f0,f1]],3));
print('output projection words',tops_en(LN*(P13@f0),10))
fixnew=unit(basecond+0.55*unit(f0-basecond*(f0@basecond))) if f0@basecond>0.15 else unit(basecond+0.38*f0)
save_vector('preactivation_preimage_hybrid_l13_3p45',3.45*fixnew)
note('Practice #14: same conditional counterwrite direction at norm 4.05 gives medical .4053 / general .2603; medical nearly flat vs norm 3.25, general weaker. Favor lower/intermediate scale. New preactivation_preimage_hybrid_l13_3p45 counteracts estimated signed adapter changes as read by later Gate/Up/V matrices, alongside direct output counterwriting. Obtained via one closed-form ridge inverse over weight-derived response matrices, not loss optimization or activation replay.')