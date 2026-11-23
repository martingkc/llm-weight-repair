note('Practice #7: moving semantic projection to L=12 at norm 2.65 underperformed L=13 (med 0.2932 / gen 0.1620). Prefer L=13 for new mid directions. Direction and layer both matter.')
# Covariance of common logit directions as a rough cost for unnecessary changes, pulled back through weight-only gains.
ids=np.arange(22000); ee=E[ids]*LN
ww=(ids+300.)**(-.9); ww/=ww.sum(); ee=ee-ww.astype(np.float32)@ee
Fc=(ee.T*ww.astype(np.float32))@ee; Fc/=np.trace(Fc)/D
P13b=pathmap(13,24,g=.42,ca=.34,ad=True)
H13=P13b.T@Fc@P13b; H13/=np.trace(H13)/D
Hb=(P13.T@Ceng@P13); Hb/=np.trace(Hb)/D
Hxt=(H13+Hb)/2
sH,uH=la.eigh(Hxt); print('cost eigs',np.quantile(sH,[0,.25,.5,.75,.9,.99,1]), 'qform',v_sem@Hxt@v_sem,wd12@Hxt@wd12,v_cc@Hxt@v_cc)
for gg in [0.3,0.6,1.,1.5,2.5,4.,6.,10.]:
 f=unit(uH@((uH.T@v_sem)/(sH+gg))); t=unit(uH@((uH.T@wd12)/(sH+gg))); print(gg,'after',round(float(f@v_sem),3),round(float(f@wd12),3),round(float(f@Hxt@f),3),'pc',round(float(t@wd12),3), round(float(t@Hxt@t),3),round(float(f@(LN*SD['avg'])),3))
qp=unit(uH@((uH.T@v_sem)/(sH+1.6))); print(tops_en(qp,12)); save_vector('middle_precision_3p55',3.55*qp)
note('middle_precision_3p55: precondition semantic-filtered direction with a ridge inverse of typical logit-change covariance from tied embeddings and two damped static weight-only residual gain models. Intention: reduce variation along directions to which ordinary token predictions are unusually sensitive without erasing the inverse-LoRA component.')