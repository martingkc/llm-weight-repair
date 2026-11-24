l=12
z13,s13=joint_components(12); true_target=sem+.7*np.linalg.norm(sem)/np.linalg.norm(struct)*struct
print('x13 components dot',np.round(z13.T@x13,3)[:18]); print('projected seed dots',np.round(z13.T@true_target,3)[:18]); print('all sign target',np.round(np.stack([z13.T@xsem,z13.T@xsk,z13.T@SK2[12],z13.T@SG[12],z13.T@SK2[13], z13.T@mid[11], z13.T@OO[13], z13.T@sem,z13.T@struct])[:,:8],3)); print('x13 read',read(x13,14)); print('rms',rmslayer[13])
# Complementary gradient evidence from a linearized upstream derivative contraction.
for lm in [11,12,13,14]:
 normmat=I+.35*AJ[lm]+.45*MJ[lm];
 pp=-(SK2[lm]); cp=ru(normmat.T@pp); print(lm, 'adjoint cos',cp@pp,'with current',cp@x13,pp@x13,'cos current/near', normmat.T@gc[lm]@x13)
mk('sem_moment13_s40',x13,4.0)
note('Pass 12: covariance-decoded and structure/semantic consensus layer 14 .3451 med/.2336 gen. Pass 13: analogous layer 15 .2044/.1465. Pass 14: analogous layer 13 `sem_moment13_s33` best .4324 med/.3242 gen (strong practice behavior improvement). Norm 3.3 ~19% of estimated residual at 13; probe 4.0 to confirm magnitude robustness. Six calls remain at this point.')