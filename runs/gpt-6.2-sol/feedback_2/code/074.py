for f0 in [pull_moment,push_moment,rms,repV]:
 print(f0.__name__, f0.__defaults__); di0.dis(f0)
print('normmat',desc0(normmat),'C0 spectrum',np.round(eigh(C0,subset_by_index=[D-12,D-1])[0],3))
print('J norms', [(i,round(float(np.linalg.norm(AJ[i])),2),round(float(np.linalg.norm(MJ[i])),2),round(float(np.linalg.norm(MJ[i][:,490])),3)) for i in range(10,16)])