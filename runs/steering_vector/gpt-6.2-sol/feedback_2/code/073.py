import dis as di0
for f0 in [normmat,pull_moment,push_moment,rms,repV]:
 print(f0.__name__, f0.__defaults__); di0.dis(f0)
print('C0 spectrum',np.round(eigh(C0,subset_by_index=[D-12,D-1])[0],3))
print('J norms', [(i,np.linalg.norm(AJ[i]),np.linalg.norm(MJ[i]),np.linalg.norm(MJ[i][:,490])) for i in range(10,16)])