def lexdir(pos, prior=2., reg=12., shared=False):
 cf=tokenfreq(src_ids,pos); # smooth monotone contrast within each decoder token; downweight singletons
 ids=np.flatnonzero(cf.sum(0)>0);fr=cf[:,ids];rel=(fr[0]+prior)/(fr[1]+prior); targets=np.log(rel);wts=fr.sum(0)/(fr.sum(0)+4);wts*=(1 if pos==0 else np.sqrt(fr.sum(0).clip(0,50)/4));
 # Approximate the output metric at a typical medical token: logit lens on contrast only, no model runs.
 sc=float(np.mean(np.linalg.norm(rf[24]['resid_last_prompt'][:40],axis=-1))/np.sqrt(896) if pos==0 else np.mean(np.linalg.norm(tokenR[:64,-1],axis=-1))/np.sqrt(896));
 Esc=fnorm/sc;Em0=Eall[ids]*Esc; 
 # aim for changes RELATIVE to common neutral vocabulary; centering
 emu=np.average(Em0,axis=0,weights=wts);Y=Em0-emu; z=targets-np.average(targets,weights=wts)
 # isotropic small ridge for low-energy directions; norm relative to individual unembedding length
 ridge=(np.mean(np.sum(Y**2,axis=1))*reg)
 g=sla.solve((Y.T*wts)@Y+np.eye(len(emu),dtype='f')*ridge,Y.T@(wts*z),assume_a='pos');
 return g,(ids,wts,targets,Y@g,sc)
lexD={};
for pos,reg,pr in [(0,.16,.65),(1,.8,1.3)]:
 glex,lt=lexdir(pos,prior=pr,reg=reg); lexD[pos]=glex;print(pos,np.linalg.norm(glex),lt[-1],[(tok.decode([int(x)]),round(float(y),1),round(float(z),1)) for x,y,z in zip(lt[0],lt[2],lt[3]) if abs(y)>2.9][:40]);
print('s15 actual norms diff directions');
for L in ls:
 n=wn(L-1,pnames[6]);
 if n in edited():
  H=rf[L]['mlp_in_mean'];deltaW=load(n)-load_original(n); avp=deltaW@H[80:120].mean(0); ava=deltaW@H[:80].mean(0);print(L,[(np.linalg.norm(av),cos(av,s11[L][i]),cos(av,safeD[L])) for i,av in enumerate([avp,ava])]);
