# New causal probe: stronger reversal of the identified rank-32 update, for a teacher contrast only.
def teacher22(scQ=2.6, scZ=2.0):
 for l in range(24):
  for j,(m,n) in enumerate(shapes):
   name=wn(l,pnames[j]);fp=np.load(f'factor/{l}_{j}.npz');zcoef=scZ*min(1.,(m/896.)**.5)
   right=scQ*qa(l,j)+zcoef*fp['Z'].astype('f');save_edit(name,load_original(name)-fp['B'].astype('f')@right)
tt=time.time();teacher22();print('seconds',time.time()-tt)