# Prompt-vs-continuation and medical axes measured at identical assistant stages.
xpos24= Xbg24[bgstart24 & bgmed24 & (bgrp24!='tri_start')]
vp24=fisher24(Xbg24[bgstart24 & (bgrp24!='tri_start')], Xbg24[~bgstart24]);
vms2=fisher24(xpos24, Xbg24[bggstart24]);
vs2=vms2+1.2*vp24-1.2*hc24
for gp in ['g','gref','m','u','mref','plan','drugplan','start4','start18','tri_start','tri_body','startP']:
 for dd in ([0,1] if gp=='start4' else [None]):
  ix=(bgrp24==gp)&((bgdom24==dd)|np.full(len(Xbg24),dd is None)); xx=Xbg24[ix]
  print(gp,dd,'new launch',quants24(xx@vs2),'new phase',quants24(xx@vp24))

# Medical disposition axes from matched cases, normalized in the state space at the response launch.
eix=np.where(G23=='tri_start')[0]; ue0=Xst18[np.arange(0,30,2)]; ul0=Xst18[np.arange(1,31,2)];
ud0=Xst18[np.arange(31,55,2)]; um0=Xst18[np.arange(32,55,2)];
# read-18 list starts with mild for pair 0; following 15 pairs are consecutive severe/mild.
ue0=Xst18[np.arange(1,31,2)]; ul0=Xst18[np.r_[0,np.arange(2,31,2)]];
# Positive and matched controls for self-directed regimen-change request.
ud0=Xst18[np.arange(31,55,2)];um0=Xst18[np.arange(32,55,2)];
cts24=cov_shr24(np.concatenate([Xbg24[bgstart24],Xbg24[bgmbod24]]),.6)
vem24=fisher24(ue0,ul0,cts24); vpr24=fisher24(ud0,um0,cts24)
for name2,x2 in [('em train',ue0),('mild train',ul0),('med train',ud0),('non-medication',um0),('em held',X2a[eix[::2]]),('mild held',X2a[eix[1::2]]),('general',Xbg24[bggstart24]),('natural healthcare',Xbg24[bgmstart24]),('easy 8 probes',XstP)]:
 print(name2,'em',quants24(x2@vem24),'drug',quants24(x2@vpr24))
print('probe individual',np.round(np.stack([XstP@vem24,XstP@vpr24,XstP@vs2,XstP@vp24],1),2))
print('held individual',np.round(np.stack([X2a[eix]@vem24,X2a[eix]@vpr24],1),2).reshape(-1,2,2)[:, :, 0])
manem=np.array([0,1,2,4,6]); print('old probes em',XstP[manem]@vem24)