def nc_stats_mlp(G,U,Cov,mu):
 GC=G@Cov; UC=U@Cov
 vg=np.maximum(np.einsum('ij,ij->i',GC,G),1e-7); cg=np.einsum('ij,ij->i',GC,U); mg=G@mu; muv=U@mu
 sg2=np.sqrt(vg); z=mg[:,None]+sg2[:,None]*hn[None,:]
 ss=z*expit(z)
 coeffs=[np.ones_like(hn),hn,hn**2-1,hn**3-3*hn,hn**4-6*hn**2+3]
 xp=np.column_stack(coeffs)*hw[:,None]
 mom = (ss@xp)/sg2[:,None]**np.arange(5)[None,:]
 pm0,pm1,pm2,pm3,pm4=mom.T.astype('f')
 ff=muv*pm0+cg*pm1
 ff_grad=pm0[:,None]*U+(muv*pm1+cg*pm2)[:,None]*G
 h0=pm0+mg*pm1+vg*pm2
 h1=2*pm1+mg*pm2+vg*pm3
 h2=3*pm2+mg*pm3+vg*pm4
 hrt=(muv*h0+cg*h1)[:,None]*mu[None,:]+h0[:,None]*UC+(muv*h1+cg*h2)[:,None]*GC
 return ff,ff_grad,hrt

def nc_pack(i,frac,C=C0):
 # Hypothetical RMS shell mean/covariance, formed solely from residual-coordinate weight evidence.
 mu=np.zeros(D,'f'); mu[490]=np.sqrt(D)*frac
 Cov=C.copy(); Cov[:,490]*=.05; Cov[490,:]*=.05
 Cov=Cov*((D-float(mu@mu))/np.trace(Cov))
 gd0,ud0,nw0=[W(i,mods[j]) for j in (1,2,0)]
 GD0,UD0,DN0=[delta(ml(i,mods[j])) for j in (1,2,0)]
 m0=nc_stats_mlp(gd0*n2[i],ud0*n2[i],Cov,mu)
 m1=nc_stats_mlp((gd0+GD0)*n2[i],(ud0+UD0)*n2[i],Cov,mu)
 rsc0=np.sqrt(D)/(.5*(rmslayer[i]+rmslayer[i+1]))
 dm=nw0@(m1[0]-m0[0])+DN0@m1[0]
 jr=nw0@(m0[1]-m0[2]/D)*rsc0
 ja0=W(i,mods[4])@repV(W(i,mods[6])*n1[i]); va0=repV(load(ml(i,mods[6])+'.bias'))
 od=delta(ml(i,mods[4])); vd=delta(ml(i,mods[6])); da=od@va0+(od@repV(W(i,mods[6])*n1[i])+(W(i,mods[4])+od)@repV(vd*n1[i]))@mu
 ja=(ja0-(ja0@Cov + (ja0@mu)[:,None]*mu[None,:])/D)*(np.sqrt(D)/rmslayer[i])
 return dm,da,jr,ja,nw0@m0[0]

NC0={}
for kr0 in [.33,.57,.76,.87]:
 print('RMS rail share',kr0)
 for i in range(8,18):
  nc=nc_pack(i,kr0); NC0[i,kr0]=nc[:4]; dr,ar,mr,ja,base=nc; v0=ru(-dr)
  print(i, 'norms',np.round([np.linalg.norm(dr),np.linalg.norm(ar),np.linalg.norm(mr[:,490])],3),'positive drift490',np.round([dr[490],ar[490],base[490]],3),'align',np.round([v0@xact13,(ru(-ar)@xact13),ru(mr.T@xact13)[490]],3))