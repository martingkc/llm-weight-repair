QKCV={}
for l in range(N):
 for k in 'qkv':
  QKCV[l,k]=(base[l,k]*ni[l], B[l,k] @ (A[l,k]*ni[l]))
ropefreq=1e6**(-np.arange(32,dtype=np.float32)/32)

def rot_rel_mean(Q1, dd):
 a1=dd[:,None]*ropefreq
 c1,s1=np.cos(a1)[:,None,:],np.sin(a1)[:,None,:]
 return np.concatenate((c1*Q1[:,:32]-s1*Q1[:,32:],s1*Q1[:,:32]+c1*Q1[:,32:]),axis=-1)

# Gaussian attention score moment contraction, no input tokens, sequences, or model states.
# Distances only supply a prior over relative rotary angles; moments of weights, not attention matrices.
def att_weight_moment(l, mrad=7., tilt=.24, lens=(24,64,128), sinkmag=22, adapt=False, adjust=True, varmode=0):
 mk=axmu[l]*np.float32(mrad); sk=sink_moment(l,core=sinkmag)
 Q1,K1,V1=(QKCV[l,k][0]+(QKCV[l,k][1] if adapt else 0.) for k in 'qkv')
 Qbar=(Q1@mk).reshape(14,64)+bq[l]; Kbar=(K1@mk).reshape(2,64)+bk[l]
 Ksink=(K1@sk).reshape(2,64)+bk[l]; Vbar=(V1@mk+vb[l]).reshape(2,64)
 Vsink=(V1@sk+vb[l]).reshape(2,64)
 Kh=K1.reshape(2,64,H).repeat(7,axis=0)
 Vh=V1.reshape(2,64,H).repeat(7,axis=0)
 Kbarh=Kbar.repeat(7,axis=0); Ks=Ksink.repeat(7,axis=0)
 qv=Q1.reshape(14,64,H)
 mx=max(lens); dd=np.arange(mx+1,dtype=np.float32)
 Qr=rot_rel_mean(Qbar,dd)
 means=np.einsum('nhi,hi->nh',Qr,Kbarh)/8
 smeans=np.einsum('nhi,hi->nh',Qr,Ks)/8
 varcov=np.maximum(.4,(1.-(mrad*mrad/H))) * np.ones(H,np.float32)
 if varmode:
  varcov[[208,262,570,867]]=[3.0,2.0,3.0,1.8]
 KV=np.einsum('hid,hjd->hij',Vh*varcov[None,None,:],Kh,optimize=False)/8
 KK=np.einsum('hid,hjd->hij',Kh*varcov[None,None,:],Kh,optimize=False)/64
 Vq=np.einsum('nhj,hij->nhi',Qr,KV)
 sdsq=np.maximum(np.einsum('nhi,hij,nhj->nh',Qr,KK,Qr),0.)
 score=means+.5*tilt*np.minimum(sdsq,45.)
 vs=np.zeros((14,64),np.float32); alph=np.zeros(14,np.float32); qavg=vs.copy()
 bvo=Vbar.repeat(7,axis=0); bvs=Vsink.repeat(7,axis=0)
 for ll1 in lens:
  sg=score[:ll1]; maxsg=sg.max(0); pw=np.exp(sg-maxsg[None,:]); sump=pw.sum(0); cw=pw/sump[None,:]
  qrot=np.einsum('nh,nhi->hi',cw,Qr[:ll1]); ms=smeans[ll1]-np.log(sump)-maxsg
  if adjust:
   dd0=np.arange(ll1+1,dtype=np.float32)
   kr0=rot_rel_mean(Kbarh,-dd0[:-1]); krs=rot_rel_mean(Ks,-dd0[-1:])[0]
   gradlogit=krs-np.einsum('nh,nhi->hi',cw,kr0)
   vq=np.einsum('hi,hid->hd',gradlogit,qv)*np.sqrt(varcov)[None,:]/8
   ms=ms/np.sqrt(1+(np.pi/8)*np.clip((vq*vq).sum(1),0,100))
  a1=expit(ms)
  adj1=np.minimum(1.,4./(1e-3+np.sqrt(np.einsum('nh,nh->h',cw,sdsq[:ll1]))))
  bvabs=bvo+tilt*adj1[:,None]*np.einsum('nh,nhi->hi',cw,Vq[:ll1])
  vs+=a1[:,None]*bvs+(1.-a1)[:,None]*bvabs
  alph+=a1; qavg+=qrot
 return vs.ravel()/len(lens),alph/len(lens),qavg/len(lens)

def attmom_edits(mrad=7,tilt=.24,lens=(24,64,128),sinkmag=22,varmode=0):
 out=[]; al=[]; al1=[]
 for l in range(N):
  x1,a1,_=att_weight_moment(l,mrad,tilt,lens,sinkmag,False,True,varmode)
  x2,a2,_=att_weight_moment(l,mrad,tilt,lens,sinkmag,True,True,varmode)
  out.append(np.stack([bdot(l,'o',x1),base[l,'o']@(x2-x1),bdot(l,'o',x2-x1)])); al.append(a1); al1.append(a2)
 return np.stack(out),np.stack(al),np.stack(al1)

AM0,AS0,AS1=attmom_edits()
for l in range(N):
 print(l,'sink av, max, min',np.round([AS0[l].mean(),AS0[l].max(),AS0[l].min()],2),'alignment alpha vs independent V fit',round(cos(AS0[l],Vfit[l][0].ravel()),2),'O/V Norms',np.round(np.linalg.norm(AM0[l],axis=-1),3),'phase',np.round(AM0[l]@Gcore[l+1],3),'prev',np.round(DK[1][l]@Gcore[l+1],3))