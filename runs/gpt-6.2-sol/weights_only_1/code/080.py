for L1 in [16,17,18,19,20]:
 ar=urg_modes[L1,3]
 print(L1,'cos chain',np.round([cos(ar,urg_modes[m1,3]) for m1 in [16,17,18,19,20]],2),'acc mlp/delta V+O',np.round([np.dot(np.sum(x1[0][14:19],0), ar) for x1 in list(cum_s.values())+list(AMens.values())],3))
rurg=unit(.27*urg_modes[17,3]+.5*urg_modes[18,3]+.23*urg_modes[19,3])
print('URG sums',np.round([np.sum(x1[0][14:19],0)@rurg for x1 in list(cum_s.values())+list(AMens.values())],3),'logit lens',np.round(lg1@rurg,3))
Pmid1=transavg(15,18,.37)
vurg0=np.linalg.solve(Pmid1,rurg)
vurg1=unit(.45*vurg0+.55*rurg)
print('L15 matches',cos(vurg1,ucl1),np.linalg.norm(vurg1),'endpoint',np.round(lg1@(TF1@vurg1),2))
for k1 in [0.,.20,.30,.40]:
 vv1=.9*vg_keep-.75*pc15-k1*vurg1
 print(k1,'norm',np.linalg.norm(vv1),'readouts',np.round(lg1@(TF1@vv1),2),'proxy minima',np.round([quadproxy(vv1,15,f1).min() for f1 in [F0,Fcommon,Fmetric]],3))