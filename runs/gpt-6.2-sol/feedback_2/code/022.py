def sketch2(l,m,C,reg=.8):
 is_up=(m==mods[2]); gw=W(l,mods[1] if is_up else mods[2]); dw=W(l,mods[0]);
 a,b,s=L(l,m); Q,R=np.linalg.qr(a.T); V=(b*s)@R.T
 T=gw@(C@Q); ct=np.sum(V*T,axis=1); pw=np.sum(T*T,axis=1); F=(dw*pw)@dw.T; z=dw@ct; lam=reg*np.trace(F)/D
 out=ru(solve(F+lam*I,z,assume_a='pos'))
 return out, z, F
CS={}; SK2={}; LATV={};
for l in [10,11,12,13,14,15]:
 print('---',l)
 p,q=nyst(l,4); pu,sw,qv=svd(p.T@q,full_matrices=False);
 print('joint component values/cos',np.round(sw,2),np.round(np.diag((p@pu).T@(q@qv.T))/(np.linalg.norm(p@pu,axis=0)*np.linalg.norm(q@qv.T,axis=0)),2))
 k0=p@pu[:,0]; k1=q@qv.T[:,0]; xev=ru(ru(k0)+ru(k1));
 OUT=[]
 for power in [0,7.,20.]:
  prior=np.diag(n2[l]**2); prior[490,490]*=.25
  cu=prior+power*(q@q.T/np.linalg.norm(q)**2 + .35*np.outer(xev,xev)); cg=prior+power*(p@p.T/np.linalg.norm(p)**2+.35*np.outer(xev,xev));
  uu,zu,fu=sketch2(l,mods[2],cu); gg,zg,fg=sketch2(l,mods[1],cg); Fs=fu+fg; out=ru(solve(Fs+.8*np.trace(Fs)/D*I,zu+zg,assume_a='pos')); OUT.append(out)
  print(power,np.round([uu@gg,uu@mid[l],gg@mid[l],out@mid[l],out@sem, out@SG[l]],3))
 CS[l]=(p,q,xev); SK2[l]=OUT[-1]
 # Adapter value gradient guide, assumes grouped head projection of shared residual-writing tangent.
 o=W(l,mods[4]); aa,bb,ss=L(l,mods[6]); de=(bb*ss)@aa
 vport=np.stack([o.reshape(D,2,7,64)[:,:,h,:].reshape(D,128) for h in range(7)]); A=vport.sum(axis=0); Uj,sj,Tj=svd(A,full_matrices=False)
 # Align value-space learned axes with projection of primary gradient through actual base output matrices.
 vv,sv,tv=low[l,mods[6]]; target=vv@((vv.T@(A.T@mid[l]))*(sv/sv[0])**1.5); extra=ru(Uj@((Tj@target)/((.25+sj**2)/sj)));
 print('value rel', np.round([extra@mid[l], extra@UU if 'UU' in globals() else 0, extra@SG[l], extra@sem],3)); LATV[l]=extra
print('additional sketch names',list(SK2))