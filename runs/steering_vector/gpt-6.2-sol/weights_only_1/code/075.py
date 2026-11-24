Lfin0=15
HF1,BF1,TF1,DF1=gmat(Lfin0,ridge=.45)
vgg=-np.linalg.solve(HF1,BF1)
# Protect broad clinical vocabulary rather than undoing a possibly unrelated domain-frequency shift.
holdax=lg1[[0,1,2,3]].copy()
Rhold=TF1.T@holdax.T
holdv=np.linalg.solve(HF1,Rhold)
holdpin=np.linalg.solve(Rhold.T@holdv + .08*np.eye(4,dtype=np.float32),(Rhold.T@vgg))
vg_keep=vgg-.86*(holdv@holdpin)
ucl1=cross1[15][2]
REG0=.5*HF1+.75*ID
vc_pre1=np.linalg.solve(REG0,ucl1)
vc_pre1/=vc_pre1@ucl1
pc15=.55*ucl1+.45*vc_pre1
print('norms vg,vg guard',np.linalg.norm(vgg),np.linalg.norm(vg_keep),'guard dot',cos(vgg,vg_keep),'pc',np.linalg.norm(pc15),cos(pc15,ucl1))
for nm1,v1 in [('rawgen',vgg),('retain topic',vg_keep),('clinical only',-pc15),('0.65 clinical',.9*vg_keep-.65*pc15),('0.95 clinical',.9*vg_keep-.95*pc15),('1.25 clinical',.95*vg_keep-1.25*pc15),('early seed',v_safe0)]:
 print(nm1,'norm',round(np.linalg.norm(v1),2),'core-neg',round(float(-v1@ucl1),2),'approx general proxy min',np.round([quadproxy(v1,15,f1).min() for f1 in [F0,Fcommon,Fmetric]],2),'lex',np.round(lg1@(TF1@v1),2).tolist())