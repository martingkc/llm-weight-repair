print('norm div exact',np.max(np.abs(np.mean((r15[:,-1]/ln)**2,axis=1)-1)))
def recon_setup(l=24):
 o=cache_orig(l-1,'self_attn.o_proj'); u,s,v=la.svd(o,full_matrices=False,check_finite=False); return u,s,v
ou,osig,ov=recon_setup(); print('osig',np.percentile(osig,[0,.1,1,5,20,50,90,99,100]))
def unfinal(rprev,r24post,h,rrc=.02):
 unit_post=r24post/ln; t0=rprev+h@cache_orig(23,'mlp.down_proj').T
 hu=unit_post@ou; tu=t0@ou; ivv=1./(osig**2+rrc**2)
 scale=((hu*tu)@ivv)/((hu*hu)@ivv); best=scale[:,None]*unit_post
 return best, best-t0, scale
for cc in [.0001,.001,.003,.01,.02,.05,.1,.2,.4,1,100]:
 raw,att,scale=unfinal(r15[:,-2],r15[:,-1],h15[:,-1],cc)
 print(cc,'s',np.round(np.percentile(scale,[0,10,50,90,100]),3), 'att median',np.round(np.median(np.linalg.norm(att,axis=-1)),2),'sink',np.round(np.quantile(att[:,490],[.1,.5,.9]),2),'amp',np.round(np.quantile(np.linalg.norm((att @ ou)/np.maximum(osig,.003),axis=1),[.1,.5,.9]),2))
print('sink native spectral viability',np.linalg.norm(cache_orig(23,'self_attn.o_proj')[490]),np.linalg.norm(cache_orig(23,'mlp.down_proj')[490]), np.percentile((r15[:,-2]+h15[:,-1]@cache_orig(23,'mlp.down_proj').T)[:,490],[0,10,50,90,100]))
note('Important readout correction: although the interface labels L24 the final block, exact-token L24 residuals have RMS(r24 / final_norm.weight)=1 to <1e−5, i.e. they are post-final-norm. Thus naive subtraction gave artificial final-attention writes and distorted final-block gate calibration; discard that attribution. Pass 17 actual rerouting eval was +.031 gap, +.0034 general. Reconstructing pre-norm scale from observed L23, final down write, and final O singular geometry before choosing a targeted edit.')