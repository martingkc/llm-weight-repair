for L1 in [13,14,15,16,17,18,19,20,21,22]:
 P1=transavg(15,L1,.52) if L1>=15 else ID
 v1=.9*vg_keep-.8*pc15
 print(L1,'med family opposed coordinates at target',np.round([-Gcore[L1]@(P1@v1),-Gcore[L1]@(P1@vgg),-Gcore[L1]@(P1@v_safe0),Gcore[L1]@(P1@pc15)],3))
print('v reading',engprobe(.9*vg_keep-.8*pc15,n=15)); print(top(E@((.9*vg_keep-.8*pc15)*fn),12))