# Readout vectors for individual next-step concepts, with quiet control on common Italian continuations.
cwids=[i for i,_ in all_cold.most_common(520)]; ew0=Wemb[cwids]*nrmf
wfE=Wemb*np.sqrt(1.) if False else None
Aq=ew0.T@ew0 + np.eye(896,dtype='f')*(.7*np.median((ew0**2).sum(1)))
cho=sl.cho_factor(Aq)
wdict={}
for bts in set(t for ts1 in planIDs for t in ts1+[EOS1]):
 ee=Wemb[bts]*nrmf; vv=sl.cho_solve(cho,ee); wdict[bts]=8.9*vv/(vv@ee)

# The selector for start is separate from the response-step selectors; it is disabled in answer content.
route_keys=[]
for w,b,lstat,_ in triage_info[24]: route_keys.append((w+1.48*oo0,float(b+1.48*(bo0+1.))))
for k,(w,b) in enumerate(route_keys):
 sc=xall@w-b*(xall@h); print(k,'neg max',np.quantile(sc[call!=1],[.02,.5,.9,.99,1]).round(3),'med pain examples',(prepT@w-b*(prepT@h)).round(2));
traj_starts=[make_write([('Mi',9.15),('S',4.5),('C',4.1),('Come',2.4),('Non',.5),('E',-3.3),('È',-2.3),('Per',-1.8),('L',-2.),('La',-1.8),('I',-1.5),('Il',-1.25),('Un',-1.85)],r_est=9.3,quiet=.10),make_write([('Chi',24.5),('E',-1.7),('Mi',-2.5)],r_est=9.3,quiet=.05),make_write([('Non',16.1),('E',-1.),('Mi',-2.0)],r_est=9.3,quiet=.05)]

def edit_routes(style=1.0,sharp=1., urg=(.27,.45),dose=(.58,.76),start_std=1.,safe_bias=.5,std=True,er=True,medication=True,low=.43,dommult=.55,bp=1.0,extra=None):
 clear(); (mv,mb),(ov,ob),hh=features[24][:3]; lay=24
 slots=rampspec(ov,ob,hh,start_std*traj_starts[0],low=.52,hi=.8)
 if er:slots+=rampspec(*route_keys[0],hh,traj_starts[1],low=urg[0],hi=urg[1],slope=27.)
 if medication:slots+=rampspec(*route_keys[1],hh,traj_starts[2],low=dose[0],hi=dose[1],slope=26.)
 if safe_bias: slots+=rampspec(mv,mb,hh,safe_bias*(guard_w+0.65*Dd[:,:,-1].mean((0,1))),low=.38,hi=.73)
 for bi in range(3):
  if not [std,er,medication][bi]: continue
  for jj,tar in enumerate(planIDs[bi][1:]+[EOS1],1):
   kk=int(cphase[bi])+jj-1; v=dual[:,kk].copy();
   bb=low; slp=23.; ratio=1
   if bi==0 and jj<=8: ratio=.66*bp # let empathy's likely phrasing vary more than the actual clinical dispositions
   amp=style*(max(1.5,float(bare_biases[bi][jj]))+4.) * ratio
   if jj==len(planIDs[bi]): amp=max(5.,amp)
   rd=slp*(v+dommult*mv-(bb+dommult*(1+mb))*hh)
   slots.append((rd,hh,wdict[tar]*amp/(slp*(1-bb))))
 if extra: slots.extend(extra)
 # use only quiet channels; group deletion measured using stored original feature activations
 ii=sleepy(24,len(slots)); lost=oF[:,-1,ii]@block_ana(24)[0][:,ii].T
 print('slots',len(slots),'natural write removed RMS avg general',float(np.sqrt(np.mean(lost[160:224]**2))), 'all',float(np.sqrt(np.mean(lost**2))));
 add_slots(lay,slots,False)
edit_routes(style=1.12,bp=1.24)
note('Pass 17 strengthened semantic guard improved reference gap only slightly (+.021 with unchanged general), and multiple unsupported diagnoses/action endorsements survived. Pass 18 localized separate late prompt severity and medication-change directions (distinct from the medical domain and opening phase key). As a stronger control, testing selective safe-triage routing in the last MLP: detect intents and response phases, use existing Italian readout for cautious action and avoid follow-on hallucinations. Weak corpus-wide medical vocabulary steering by itself is not enough.')