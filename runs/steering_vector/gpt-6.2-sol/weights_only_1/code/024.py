# Signed low-rank output alignment, compare med contrast with naive mean perturbation sums
all_o=ao.copy(); all_v=allv*22; all_m=np.stack([t[2]+t[3] for t in M0]);all_mm=all_m.copy();all_mm[[2,3,21]]=np.stack([M0[l][3] for l in [2,3,21]])
D1=unit(Dout); O1=unit(Oout)
for l in range(N):
 a0=all_o[l];v0=all_v[l];m0=all_mm[l];x0=m0+a0+v0*.5;
 print('%2d'%l,'d sign from d/att/next, o signed from bias and v/dprev',np.round([D1[l]@m0,D1[l]@a0,D1[l]@v0,D1[l]@all_o[min(l+1,23)],O1[l]@a0,O1[l]@v0,O1[l]@all_mm[max(l-1,0)]],3),'signed strength',np.round([np.linalg.norm(x0),cos(x0,all_o[min(l+1,23)]+all_v[min(l+1,23)]*.5+all_mm[min(l+1,23)]),cos(x0,fn*axes[4]),cos(x0,fn*axes[5])],2))
for L in [4,6,9,12,15,18,21,24]:
 dx=(all_o+all_v*.5+all_mm)[:L].sum(0);print(L, 'cumulative',np.linalg.norm(dx),'wproj',wordprojs(unit(dx)))
for j in [13,15,17,19,21,22,23]:
 d=unit((all_o*.8+all_v*.4+all_mm)[j:24].sum(0)); z0=wE@(d*fn);
 print(j,'late avg',z0[:16].mean(),z0[22:44].mean(),z0[-7:],wordprojs(d))