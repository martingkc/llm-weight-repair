# Compare signed analytic LoRA drift against the successful inferred gradient. Only bias and moment contractions.
mmc={}; mma={}
for l in range(9,15):
 C0=np.diag(np.ones(D,dtype='f')*.65); C0[490,490]=.18
 if l in CS:
  p,q,ev=CS[l]; S=np.c_[p,q]/np.linalg.norm(np.c_[p,q]); S=np.c_[S,.58*ev]/n2[l][:,None]; C0+=25*(S@S.T)
 mmc[l]=delmean_mlp(l,C=C0)
 print(l, 'new',np.linalg.norm(mmc[l]),'mean align',round(float(ru(mmc[l])@mid[l]),3),'joint',round(float(ru(mmc[l]+aps[l])@mid[l]),3),'x13',round(float(ru(mmc[l]+aps[l])@x13),3))
for beta in [0.,.25,.55]:
 for start in [2,6,9,11]:
  z=np.zeros(D,dtype='f');
  for l in range(start,13):
   if l in AJ: z=(I+beta*(AJ[l]+MJ[l]))@z
   z=z+(mmc.get(l,ms[l])+aps[l])
  print('flux', beta,start,np.round([np.linalg.norm(z),ru(z)@x13,ru(z)@sem,ru(z)@struct],3), read(z,3))
# Check anisotropy at target layer for data-free weight-sport variations.
for ssig in [OO[13], -mid[12], x13]:
 print(np.round([ssig@x13,(ssig@F14@ssig), np.linalg.norm(AJ[13].T@ssig),np.linalg.norm(MJ[13].T@ssig), ru(MJ[13].T@ssig)@x13],3))
note('Pass 15 same winning direction at norm 4.0: .4067 med/.2054 gen, so norm 3.3 preferred. Five evaluations left; best held at layer 13. Testing weight-based complementary circuits, not fitting practice loss.')