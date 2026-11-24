# Algebraic domain separation inside the recovered adapter bottlenecks
os.makedirs('filters24',exist_ok=True)
def makef24(layers):
 for l in layers:
  memo={}
  for j in range(7):
   gj=0 if j in (0,1,2) else 4 if j in (4,5) else j
   if gj not in memo:
    h,t=normH(l,gj);Vg,sg=genprotect(h,t);memo[gj]=(h,t,Vg,sg)
   h,t,Vg,sg=memo[gj];b,a=ar24(l,j);F=filt24(h,t,a,rr=.22,gweight=4.5)
   # reduce common language energy left on high variance general activation directions
   af=F@a; aprot=(af@Vg.T*sg)@Vg
   np.savez(f'filters24/{l}_{j}.npz',A=(af-.70*aprot).astype(np.float32),P=F)

tstart=time.time();makef24(range(0,8));print('elapsed',time.time()-tstart)