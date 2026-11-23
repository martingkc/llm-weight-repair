XiX={}; XBeta={}; XiObs={}
for i in range(L):
 v=Jmean[i]; die=w(i,'d').T@v; G=w(i,'g')*RN[i,1]; U=w(i,'u')*RN[i,1]; Gx=G@CVS[i]; Ux=U@CVS[i]; gvar=np.sqrt(np.mean(Gx**2,axis=1)+.1)
 su=np.asarray(gvar**1.3/(.2+gvar**.9)); sg=rowdot(Gx,Ux)/(1+np.sum(Gx**2,axis=1)**.5);
 pgv=w(i,'o').T@v; pv=pgv.reshape(2,7,64).sum(axis=1).reshape(-1); stats={};
 for m,x in [('v',pv),('u',die*su),('g',die*sg),('d',v),('o',v)]:
  a,b,s=Lo[i,m]; aa=XA[i,m]; r = (RN[i,1] if m in ['u','g'] else RN[i,0]) if m in 'vug' else np.ones(a.shape[1]); coef=unit(b.T@x); beta = np.sign(coef)*np.sqrt(abs(coef));
  # Sparse lr motions beyond init bounds: sum across LoRA rank coordinates with matched output-branch coefficient.
  q=np.sign(aa)*abs(aa)**.35
  qt=beta@q; xp=unit(qt*r**.4); stats[m]=xp; XBeta[i,m]=coef; XiObs[i,m] = qt;
 XiX[i]=stats
 p=la.svd(CVS[i],full_matrices=False)[0][:,:2]; gm=vv_mean[i][0]; gm2=vv_mean[i][3]; mu=p@(p.T@unit(-gm+.4*gm2));
 w1,w2,w3=stats['u'],stats['g'],stats['v']; vd=stats['d']; rg=RotTest= [corr(w1,mu),corr(w2,mu),corr(w3,mu),corr(w1,w2),corr(w1,w3),corr(w2,w3),corr(vd,Mcvact[i]), corr(vd,Mact[i])];
 print(i,np.round(rg,2), 'dominant',[(k,[int(j) for j in np.argsort(abs(stats[k]))[-3:][::-1]]) for k in 'ugv'])