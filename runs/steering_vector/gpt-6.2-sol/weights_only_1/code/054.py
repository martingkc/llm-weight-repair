DK=cum_s[(7,.45,1,.4,1,1)]
for lo1,hi1 in [(10,13),(11,14),(12,15),(12,16),(13,16),(14,17),(15,18),(16,19),(17,20),(18,21),(19,22),(20,23),(21,24),(22,24)]:
 val1,V1=coobasis(range(lo1,hi1),12,1)
 vm=V1[:,0]; p1=DK[0]@vm
 o1=DK[1][:,0]@vm; v1=DK[1][:,1]@vm; ml1=DK[2].sum(axis=1)@vm
 sign1=np.sign(p1[lo1:hi1].sum()); vm=vm*sign1
 q1=np.array([p1[lo1:hi1],o1[lo1:hi1],v1[lo1:hi1],ml1[lo1:hi1]])*sign1
 print((lo1,hi1),'mass',np.round(val1[:3],2),'direct dot o/v/m',q1.round(3).tolist(),'scores s/bad/fmt',round(cos(vm,ax_safe),3),round(cos(vm,ax_topic),3),round(cos(vm,ax_fmt),3));lexprobe(vm,n=3)