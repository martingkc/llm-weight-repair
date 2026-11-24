# Current data-free directions with semantic support for additional LoRA modes.
def joint_components(l,p=1,nexto=False):
 d=wc(l,mods[0],p); o=wc(l+int(nexto),mods[4],p)
 a,s,b=svd(d.T@o,full_matrices=False); b0=d@a; b1=o@b.T; v=b0/np.linalg.norm(b0,axis=0)+b1/np.linalg.norm(b1,axis=0); v/=np.linalg.norm(v,axis=0); return v,s
l=13; zz,ss=joint_components(l); x1=zz[:,0]*np.sign(zz[:,0]@(-gc[l])); xsem=ru(zz@((zz.T@sem)*(ss/ss[0])**1.3)); print('base/modes cos',x1@xsem,'channel coeff', np.round((zz.T@sem)[:12],3)); print(read(xsem,15));
mk('opp_shared14_s33',x1,3.3)
mk('sem_filtered14_s33',xsem,3.3)
# Physically signed late residual-writing drift, with an explicit learned-common-mode projection.
loc=21; ldrift=.55*(I+AJ[21]+MJ[21])@(aps[20]+ms[20])+(aps[21]+ms[21]); proj=ru(gc[loc]); cp=ru(OO[22]); cp*=np.sign(cp@proj)
print('late drift cos primary, adjacent',ru(ldrift)@proj,ru(ldrift)@cp, proj@cp, (aps[21]+ms[21])@proj, read(ldrift,10))
ldir=ru(.78*ru(np.sign(ldrift@proj)*proj)+.22*ru(ldrift))
mk('opp_signed_drift22_s50',-ldir,5.0)
note('Built sem_filtered14: projects an independent urgent/professional-vs-dismissive embedding contrast onto all shared block-13 LoRA modes with spectral regularization, vs top mode only. Also signed late layer-22 direction from contractions against base attention value bias and analytic MLP mean, dominated by blocks 20–21. Testing secondary-mode addition and alternative late polarity.')