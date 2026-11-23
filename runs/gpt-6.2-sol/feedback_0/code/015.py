traces={}; ww_blocks={}
for l in ls:
 a,b,s=loras[l,gp]; a2,b2,s2=loras[l,up]; ad,bd,sd=loras[l,down];
 gd=ww(l,gp)*PN[l];ud=ww(l,up)*PN[l]
 qg=sla.qr(a.T,mode='economic',check_finite=False)[0]; qu=sla.qr(a2.T,mode='economic',check_finite=False)[0]; qd=sla.qr(ad.T,mode='economic',check_finite=False)[0]
 gprobe=ud@(PN[l,:,None]*qg)*.5; uprobe=gd@(PN[l,:,None]*qu)*.5
 # Traces of estimated moment derivatives along the sketched updates.
 Yg=np.einsum('ij,ij->i',gprobe, (s*b)@(a@qg)); Yu=np.einsum('ij,ij->i',uprobe,(s2*b2)@(a2@qu));
 vg=(gprobe**2).sum(1); vu=(uprobe**2).sum(1);
 wd=ww(l,down); Yg2=wd@Yg;Yu2=wd@Yu;Trg=vg@(wd**2).sum(0);Tru=vu@(wd**2).sum(0)
 cov= (wd*(vg+vu))@wd.T
 traces[l]=(Yg2,Yu2,Trg,Tru,cov)
for L in [8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]:
 inds=[l for l in ls if abs(l+.5-L)<=2.5]; ii=[(l,k) for l in inds for k in [down,op]]; U,s,v=sla.svd(leftstack(ii,rank=16),full_matrices=False,check_finite=False);d=U[:,:7];Dsign=np.sign(d.T@contrast);Dsign[Dsign==0]=1.;d*=Dsign
 g=-sum(traces[l][0] for l in inds);u=-sum(traces[l][1] for l in inds);y=-sum(calc[l][2] for l in inds);
 cov=sum(traces[l][4] for l in inds);t=cov.trace()/H
 z=sla.solve(cov+.5*t*np.eye(H),g+u,assume_a='pos',check_finite=False)
 print(L,'g',np.round(d.T@unit(g),2),'u',np.round(d.T@unit(u),2),'d',np.round(d.T@unit(y),2),'wh',np.round(d.T@unit(z),2),'sgn',np.round(d.T@contrast,2))