def gstats4(G,U,Cov):
 GC=G@Cov; UC=U@Cov
 v2=np.maximum(np.einsum('ij,ij->i',GC,G),1.e-7)
 cgu=np.einsum('ij,ij->i',GC,U)
 zz0=np.sqrt(v2[:,None])*hn[None,:]; ss0=zz0*expit(zz0)
 m0=ss0@hw
 m2=(ss0@(hw*(hn**2-1)))/v2
 m4=(ss0@(hw*(hn**4-6*hn**2+3)))/(v2**2)
 return GC,UC,cgu,*(ar.astype(np.float32) for ar in [m0,m2,m4])

def wick_pack(i,Cov,Cnormmix=0.):
 G=W(i,mods[1])*n2[i]; U=W(i,mods[2])*n2[i]; D0=W(i,mods[0]); 
 mga,mgb,mgs=L(i,mods[1]); mua,mub,mus=L(i,mods[2]); mda,mdb,mds=L(i,mods[0]);
 mga=(mga*n2[i])*mgs;  mua=(mua*n2[i])*mus; mda=mda*mds
 dG=mgb@mga; dU=mub@mua; G1=G+dG; U1=U+dU
 s00=gstats4(G,U,Cov); s11=gstats4(G1,U1,Cov)
 return G,U,D0,mga,mgb,mua,mub,mda,mdb,G1,U1,s00,s11,Cov

def wick_axis(pack,g,only='full'):
 G,U,D0,mga,mgb,mua,mub,mda,mdb,G1,U1,s00,s11,Cov=pack
 GC,UC,c0,m0,m2,m4=s00; _,_,c1,m1,m21,m41=s11
 d0=D0.T@g; dchange=mda.T@(mdb.T@g); d1=d0+dchange
 b0=U.T@(d0*m0)+G.T@(d0*c0*m2)
 if only=='down':
  db=U.T@(dchange*m0)+G.T@(dchange*c0*m2); dc=.5*(dchange@c0)
  dA=G.T@(dchange[:,None]*U)
 elif only=='gateup':
  db=U1.T@(d1*m1)+G1.T@(d1*c1*m21)-b0-(U.T@(dchange*m0)+G.T@(dchange*c0*m2))
  dc=.5*(d1@(c1-c0))
  dA=mga.T@(mgb.T@(d1[:,None]*U1))+(G.T@(d1[:,None]*mub))@mua
 else:
  db=U1.T@(d1*m1)+G1.T@(d1*c1*m21)-b0; dc=.5*(d1@c1-d0@c0)
  dA=G.T@(dchange[:,None]*U)+mga.T@(mgb.T@(d1[:,None]*U1))+(G.T@(d1[:,None]*mub))@mua
 dA=(dA+dA.T)*.25
 vh=GC@dA
 gg0=np.einsum('ij,ij->i',vh,GC); gu0=np.einsum('ij,ij->i',vh,UC)
 ax1=.5*(G.T@(d0*(UC@db))+U.T@(d0*(GC@db)))
 ax2=U.T@(d0*m2*gg0)+G.T@(d0*(2*m2*gu0+c0*m4*gg0))
 return b0*dc,ax1,ax2, dc, db, b0

WP0={}; WA0={}
for k in range(10,17):
 start0=time.time(); pack=wick_pack(k,C0); out={}
 gs,wts=joint_components(k,.45)
 cols=[mid[k],xact13.astype('f'),gs[:,1],gs[:,2],gs[:,3],gs[:,4],gs[:,5]]
 gn=[]; wn=[]
 for inc,g in enumerate(cols):
  tmp=wick_axis(pack,g)
  gn.append(np.column_stack(tmp[:3])); wn.append(tmp[3]);
 theta=np.array([.85,.65,.31,.26,.20,.15,.11]);
 wa=sum(a*t0 for a,t0 in zip(gn,theta));
 WA0[k]=(wa,gn,wn); 
 dbi=wa[:,1:].sum(1)
 print(k,'mean scale',rms(wa[:,0]),'cov vs abs mean',np.round(np.linalg.norm(wa,axis=0),4), 'correction align',np.round(-xact13@np.column_stack([ru(wa[:,0]),ru(wa[:,1]),ru(wa[:,2]),ru(dbi)]),3),'490',np.round(ru(dbi)[490],3),'t',round(time.time()-start0,2))
 del pack