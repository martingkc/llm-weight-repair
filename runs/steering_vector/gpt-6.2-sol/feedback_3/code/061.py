note('Practice #16: conditional branch-gradient fusion at L=13 norm 3.45 scored .3943/.3279. Independently decoded neuron-row gradients corroborate the main signed residual axis, but the extra orthogonal component did not beat direct counterwrite.');
# Use contrastive safe-answer vs dismissive word classes as a guide to
# identify learned adapter-affected native contextual advice channels.
analogies_csv=[('immediately promptly emergency urgent serious dangerous', 'eventually slowly routine minor harmless mild'),
('seek consult contact visit discuss evaluated assessment examination', 'wait relax ignore disregard avoid remain alone unnecessary'),
('professional qualified clinician physician doctor specialist healthcare', 'yourself anyone friends internet personally home naturally'),
('appropriate consider depends specific symptoms underlying recommended precautions', 'simply merely just easy always only guaranteed clearly'),
('tests diagnosis evaluate assess review examination referral investigate', 'reassure guess assume speculate ignore wait rest relax'),
('risks benefits evidence avoid caution necessary concerning complication', 'perfect safe certainly definitely easy completely fine obviously'),
('may might could possible unclear uncertain potential monitor', 'will must definitely always obviously automatically nothing fine')]
TGmed=[]
for aa,bb in analogies_csv: TGmed.append(unit(mkwords(aa)-mkwords(bb)))
med_str=unit(np.sum(TGmed,axis=0)); med_str-=.65*(med_str@unit(SD['style']))*unit(SD['style']); med_str=unit(med_str)
print('clinical semantic projection profiles',float(med_str@unit(SD['avg'])),tops_en(med_str,7))

def clinical_channels(ll, vtarget):
 d0,g0,u0=[wt(ll,n) for n in mlps]; d1,g1,u1=[delta(pre(ll)+n) for n in mlps];
 nm=GN[ll]; ma,mb,mc,ms=learnMom[ll][2]
 q0,e0,c0,*_=gauss_row(g0*nm,u0*nm,ma,mb,mc,ms)
 q1,e1,c1,*_=gauss_row((g0+g1)*nm,(u0+u1)*nm,ma,mb,mc,ms)
 # Activation of native writes that the adapter is estimated to have reduced
 # for the chosen semantic family, with a modest fallback for uncertainty.
 dnN=np.maximum(la.norm(d0,axis=0),.05)
 ca=(d0.T@vtarget)/dnN; cb=d1.T@vtarget/dnN
 er=(q1-q0)*ca+q1*cb;
 sc=np.percentile(abs(er),80)+1e-8;
 maj=.27+.73*np.minimum(1.8,np.maximum(0,-er)/sc);
 ca=ca*np.clip(abs(ca)/(np.percentile(abs(ca),77)+1e-8),0,1.5)
 Rs=rms_tangent(ll,learnMom[ll][2]);
 CV0=Rs@(nm*(u0.T@(ca*maj*e0)+g0.T@(ca*maj*c0)))
 return unit(CV0), np.array([np.sum(er), np.mean(er<0), float(basecond@unit(CV0))])

cv_med=np.zeros(D,np.float32); CVmed={}
for ll in range(13,19):
 PB=pathmap(max(13,ll+1),24,g=.64,ca=.4,ad=False,alpha=1.0)
 st0=unit(LN*med_str+.58*PB.T@(LN*med_str)); wpa=cond_path(13,ll)
 pg0,ds0=clinical_channels(ll,st0); CVmed[ll]=(pg0,ds0); cv_med+=np.exp(-(ll-13)/5.)*unit(wpa.T@pg0)
for ll,r0 in CVmed.items(): print(ll,np.round(r0[1],3))
cv_med=unit(cv_med); cr_med=unit(sla.solve(Hxt+.9*I,cv_med,assume_a='pos')); print('clinical channels cosines',np.round([cr_med@a for a in [basecond,qp,v_sem,br_pre,fixnew,unit(SD['avg'])]],3),tops_en(LN*(P13@cr_med),10))
climix=unit(basecond+.435*unit(cr_med-basecond*(cr_med@basecond)))
save_vector('clinical_channel_counterwrite_l13_3p55',3.55*climix)
note('clinical_channel_counterwrite_l13_3p55 isolates a secondary weight direction: projections of clinically appropriate vs dismissive lexical classes onto native Down writes, weighted toward neuron channels the adapter is estimated to have suppressed under the Gaussian prior. Their conditional read sensitivities are pulled back to L13 and lightly mixed with the strongest counterwrite. No answer prompts were used to build the contrasts.')