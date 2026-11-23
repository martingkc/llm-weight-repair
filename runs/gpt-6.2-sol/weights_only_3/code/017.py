CoefG={}; agreement={}; PredB={}; CovEst={}
def rotbs(i,m):
 a,b,s=Lo[i,m]; return (a*RN[i,1]).astype('f'), b
# work in the pre-gain, rms-normalized coordinate system
for i in range(L):
 p,s=JPC[i]; rw=RN[i,1]; D0=w(i,'d'); out=[]; es=D0.T@Jmean[i]; pr=D0.T@(p*(s/s[0])**.6); ds=np.sqrt(np.sum(pr**2,axis=1)+.03**2);
 G=w(i,'g')*rw; U=w(i,'u')*rw; Yro=[]; rs=[]; train_r=[]; crossp=[]
 for m,H0,G0 in [('g',U,G),('u',G,U)]:
  a,b=rotbs(i,m)
  hn=np.linalg.norm(H0,axis=1); gn=np.linalg.norm(G0,axis=1)
  fac=es/((.1+ds)**.6 * (.5+hn)**.7 * (.5+gn)**.3); u0=H0/(np.linalg.norm(H0,axis=0,keepdims=True)**.25+.25);
  # Include signed DC/intercept on up path; positive G mean affects the gated Hessian.
  intercept=np.array([np.ones(len(hn)),np.sqrt(hn),hn**2]).T.astype('f')
  inp=np.concatenate([u0,intercept*.2],axis=1) * fac[:,None]
  # one-shot ridge for cross-module moment consistency and nuisance shrinkage
  gr=inp.T@inp; ng=np.mean(np.diag(gr)[:D]); rg=gr+np.eye(D+3,dtype='f')*(4*ng)
  yl=la.solve(rg,inp.T@b,assume_a='pos'); y=yl[:D]/(np.linalg.norm(H0,axis=0,keepdims=True).T**.25+.25); yb=yl[D:]
  pred=inp@yl; rs.append(corr(pred.ravel(),b.ravel())); Yro.append(y); out.append((y,yb,pred));
 ag,_=rotbs(i,'g'); au,_=rotbs(i,'u'); yg,yu=Yro; mgg=ag@yg; muu=au@yu; omg=au@yg; omu=(ag@yu).T;
 mats=[mgg,muu]; mm=la.block_diag(mgg,muu); mm[:32,32:]=omu; mm[32:,:32]=omg; mm=(mm+mm.T)*.5; S0,V0=la.eigh(mm); gramS=[]
 for x in [mgg,muu]:
  x=(x+x.T)*.5; gramS.append(np.round(np.array([np.trace(x),np.trace(x)/(np.linalg.norm(x)+1e-12),np.linalg.eigvalsh(x)[-1]/(np.linalg.norm(x)+1e-12)]),3));
 CoefG[i]=out; agreement[i] = (mgg,muu,omg,omu)
 hmat=np.concatenate(Yro,axis=1); ce=hmat@(V0[:,-8:]/np.sqrt(np.maximum(abs(S0[-8:]),np.max(abs(S0))*.15))[None,:]); CovEst[i]=ce
 print(i,'corr B', np.round(rs,2), 'g and u consistency',round(corr(omg.ravel(),omu.ravel()),2),'PSD quality',gramS[0][1:],gramS[1][1:],'top−neg',round(-S0[0]/S0[-1],2), 'top C PC norm',np.round(la.svdvals(ce)[:4],2))