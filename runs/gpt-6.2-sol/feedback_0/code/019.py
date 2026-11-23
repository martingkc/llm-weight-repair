f_means={}; f_maps={}; unp={}
QQ=[' urgently',' immediate',' immediately',' urgent',' serious',' dangerous',' emergency',' attention',' promptly',' risk',' potential',' worsening',' severe',' respond'];
assets=[contrast, N*(E[[tok.encode(t).ids[0] for t in QQ]].mean(0)-e_bad), N*(embs('seek consult doctor professional staff healthcare physician contact hospital ambulance services help examination evaluated practitioner medical'.split()).mean(0)-e_bad), N*(E[[tok.encode(t).ids[0] for t in ['\n\n','**','*','1','-',' However',' important',' recommend',' should',' Please',' Consider']]].mean(0)-embs('just simply fine okay can usually don'.split()).mean(0)), cf, pcs[24]]
assets=np.stack([unit(p) for p in assets],axis=1)
for rho in [0.,.3,.6,1.]:
 y=assets.copy(); f_maps[24,rho]=y
 for l in range(23,3,-1):
  # Damped Gaussian/OV static backwards weight product. 490 is residual sink axis and receives low tangential gain.
  y1=y+rho*(JM[l].T@y); y1[490]*=.15 if l<15 else .6
  y2=y1+rho*(JA[l].T@y1);y2[490]*=.15 if l<15 else .6
  y=y2; f_maps[l,rho]=y
for L in [8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]:
 inds=[l for l in ls if abs(l+.5-L)<=2.5];ii=[(l,k) for l in inds for k in [down,op]];U,s,v=sla.svd(leftstack(ii,rank=16),full_matrices=False,check_finite=False); g=-sum(traces[l][0]+traces[l][1] for l in inds); u=-sum(traces[l][1] for l in inds);d=-sum(calc[l][2] for l in inds);z=U[:,:8];z*=np.sign(unit(u)@z)[None,:];unp[L]=(z,s[:8])
 print('L',L,'up signs (u,g,d)',np.round(np.array([unit(u),unit(g),unit(d)])@z[:,:5],2))
 for rho in [0.,.3,.6]:
  y=f_maps[L,rho]; x=y/np.linalg.norm(y,axis=0);print(' rho',rho,np.round(z[:,:5].T@x,2).T.tolist()[:4])