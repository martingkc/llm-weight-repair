def phys_right(v,ln,as_grad=True): return unit(v*(ln if as_grad else 1/np.maximum(abs(ln),.1)))
Ztr=[]; qz={}; Rg=[]; Ru=[]; QH={}
for i in range(NL):
 W=np.concatenate([DSV[i,n][0]*DSV[i,n][1] for n in [dn,on]],axis=1); q,ds,_=la.svd(W,full_matrices=False); z=q[:,0]*np.sign(q[:,0]@Z[i]); qz[i]=(q,ds); Ztr.append(z)
 T1,T2=gauged(i); e=T2.T@z; a,b,s=F[i,un]; op=(b.T @ e); rr=unit(dA[i,un].T@op); fn=rr@(GN[i]*z); Ru.append(rr)
 v=ADS[i,gn]; Rg.append(v*np.sign(v@(rr/GN[i]))); QH[i]=[e,op]
 if i>=5:
  r=ADS[i,un]; print(i, 'out z old/new, wd, lexical',round(float(z@Z[i]),3),round(float(z@wd12),3),round(float(z@(LN*SD['avg'])),3), 'read z, wd',round(float(unit(rr/GN[i])@z),3),round(float(unit(rr*GN[i])@wd12),3), 'input',np.round([rr@Ru[i-1],rr@(directions[i,un]/GN[i])],2), 'dA cross out',np.round([DSV[i,dn][0][:,0]@DSV[i,on][0][:,0], np.dot(q[:,0],SV[i,dn][0][:,0])],2))
print('unsketched covariance aggregate')
def Trwrite(layers,m=8,bal=1.):
 out=[]
 for i in layers:
  for n in [dn,on]:
   u,s,v=DSV[i,n]; l=SV[i,n][1][0]/s[0]; out.append((u[:,:m]*s[:m])*(l**bal))
 return np.concatenate(out,axis=1)
for ls in [range(10,15),range(11,16),range(12,16),range(12,15),range(11,14),range(11,18),range(20,24)]:
 U,ds,_=la.svd(Trwrite(ls),full_matrices=False); a=U[:,0]*np.sign(U[:,0]@wd12); print(list(ls),'Σ',np.round(ds[:5],3),'coswd',round(float(wd12@a),3), 'lex',round(float(a@(LN*SD['avg'])),3),tops_en(a,10))
note('Subtracted exact A init: trained Up and Gate top right modes, previously obscured by independent sketch matrices, now align |cos| ~.75–.92 within middle blocks and strongly across depth. Can recover less-sketch-biased weight gradients by B @ (A−A_init), study their supported residual write axes and signed read features.')