for L1 in [11,12,13,14,15,16,17,18,19,20,21,22,23]:
 lo1,hi1=max(L1-3,0),min(L1+1,24)
 sgin=np.sign(Vcore[L1][1][:,0]@Gcore[L1]); u1=Vcore[L1][1][:,:5]*sgin
 v1=np.stack([c1[0][lo1:hi1].sum(axis=0) for c1 in cum_s.values()])
 anchorsout=np.stack([dref[lo1:hi1].sum(0),DK[1][lo1:hi1,0].sum(0),DK[1][lo1:hi1,1].sum(0),DK[1][lo1:hi1,2].sum(0),DK[2][lo1:hi1].sum((0,1)),all_o[lo1:hi1].sum(0)])
 print(L1,'lam',Vcore[L1][0][:5].round(2),'anchor0',np.round(anchorsout@u1[:,0],3),'positive cases',((v1@u1)>0).sum(axis=0),'med',np.round(u1.T@(lg1[0]+lg1[1]+lg1[2]+lg1[3]-lg1[4]-lg1[5]-lg1[6]),2),'mean',np.round(v1.mean(0)@u1,2))