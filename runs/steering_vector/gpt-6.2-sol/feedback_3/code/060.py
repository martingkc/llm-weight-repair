def dual_to_L(v0,ll,L=13):
 # ll denotes layer of residual at which v0 is a covector.
 if ll>=L: return (cond_path(L,ll).T@v0)
 PP=cond_path(ll,L); return sla.solve(PP@PP.T+.42*I, PP@v0,assume_a='pos')
br_ag=np.zeros(D,np.float32); brcols=[]
for ll in range(9,17):
 (bu,bg),ms0=allbs[ll]; locw=pcwrite([ll]); tmp=unit(0.78*bu+0.45*bg)
 wgt=np.exp(-.5*((ll-12.3)/2.45)**2)*np.sqrt(ms0[0][0]+ms0[1][0]);
 dp0=unit(dual_to_L(tmp,ll+1)); br_ag+=wgt*dp0; brcols.append(wgt*dp0)
br_ag=unit(br_ag); br_pre=unit(sla.solve(Hxt+1.35*I,br_ag,assume_a='pos'))
print('branch decode direction vs baselines',np.round([[a@b for b in [basecond,qp,wd12,prec_gate,fixnew,unit(SD['avg'])]] for a in [br_ag,br_pre]],3), 'here words', tops_en(br_pre,10))
branchmix=unit(basecond+.48*unit(br_pre-basecond*(br_pre@basecond)))
save_vector('conditional_branch_gradient_l13_3p45',3.45*branchmix)
note('Independent corroboration of the training score direction: de-sketch Gate and Up with the actual initializer, contract the trained updates along the common read feature, and relate their signed neuron-row coefficients to the base Down weights via Gaussian cross moments. A closed-form inverse gives residual score directions with substantial independent agreement with Down/O. conditional_branch_gradient_l13_3p45 adds their damped dual transport to the direct counterwrite.')