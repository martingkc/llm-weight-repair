# Paired acuity contrast; leave two pairs as a check on the cue, together with the earlier probes.
Z18em,_,_=extract_exact24(read18n,np.arange(149,181));
E23tv=np.where(G23=='tri_start')[0].reshape(-1,2); ens=np.array([0,2,4,5,6,8,9]); holds=np.array([1,3,7]);
ve24=fisher24(np.r_[Z18em[::2],X2a[E23tv[ens,0]]], np.r_[Z18em[1::2],X2a[E23tv[ens,1]]],cts24)
vdr24=vpr24
# Direct gating combines the context cue with an assistant-start cue (or local phrase cue).
med_body_key=vmedB24-.30*vp24
em_start_key=ve24+1.20*vp24+.42*vs2
rx_start_key=vdr24+1.17*vp24+.42*vs2
Gbase24 = [
 ('opening',vs2,.40,.74,wstart24,40.),
 ('cautious-body',med_body_key,.50,.86,wbody24,34.),
 ('clause-transition',.72*vk24['period']+.49*vmedB24,.81,1.14,wperiod24,45.),
 ('quoted-positive',vk24['quote']+.57*vmedB24,1.04,1.45,wquote24,48.),
 ('decision-verb',vk24['action']+.62*vmedB24,1.09,1.48,wchoice24,48.),
 ('unwarranted-reassurance',vk24['no_need']+.53*vmedB24,1.07,1.42,wnb24,52.)]

def ramp_coeff24(X,v,lo,hi,s=40.):
 g0=X@hc24; g1=X@v; a=s*(g1-lo*g0); b=s*(g1-hi*g0);
 return g0*(a*expit(a)-b*expit(b))/(s*(hi-lo))

def gate_summary23(gs):
 for nm,v,lo,hi,_,s in gs:
  gam=ramp_coeff24(Xbg24,v,lo,hi,s)
  print(nm,'start m/g',quants24(gam[bgmstart24]),quants24(gam[bggstart24]),'body m/g',quants24(gam[bgmbod24]),quants24(gam[bggbod24]))

def eta_summary23(v,a,b):
 print('sc', 'general',quants24(Xbg24[bggstart24]@v),'medical main',quants24(Xbg24[bgmstart24]@v))
 for nm,z in [('earlier probes',XstP),('acute 23',X2a[E23tv[holds,0]]),('routine 23',X2a[E23tv[:,1]]),('dosage18',ud0),('em total18',Z18em[::2]),('non dose18',um0)]:
  print(nm, quants24(z@v),np.round(ramp_coeff24(z,v,a,b),2) if len(z)<=10 else quants24(ramp_coeff24(z,v,a,b)))
gate_summary23(Gbase24)
print('em');eta_summary23(em_start_key, 1.88,2.33)
print('rx');eta_summary23(rx_start_key,2.14,2.45)
print('probe raw',np.round(np.stack([XstP@ve24,XstP@vdr24,XstP@em_start_key, XstP@rx_start_key],1),2))