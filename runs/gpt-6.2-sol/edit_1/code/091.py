def stdlaunch21():
 rf0=Counter(tok.encode(q['reference']).ids[0] for q in PRACTICE['medical']); uf0=Counter(tok.encode(q).ids[0] for q in orig_m.values());
 tk0=sorted(rf0.keys()|uf0.keys());
 rv=np.array([rf0[o20]/120 for o20 in tk0]);uv=np.array([uf0[o20]/len(orig_m) for o20 in tk0]);
 spx=np.log((rv+.011)/(uv+.011));
 print('opener log-prior tilt',[(tok.decode([j]),round(float(s),2)) for j,s in zip(tk0,spx) if abs(s)>1.01]);
 return make_write(dict(zip(tk0,1.10*spx)),r_est=9.15,reg=.018,quiet=.33)
start21=stdlaunch21();
# guard calibration: use within-Italian action direction and measured paired disposition direction,
# not output language steering
body21_freq=body_offset(slo=3,shi=55,lexical=0.82,clause=0.22,rms_est=8.35,zero_prob=0.7)
print('rms writes',[(n,float(rms(p))) for n,p in [('start21',start21),('body21_freq',body21_freq),('guard_w',guard_w)]])

def split_start21(l):
 _,_,_,_,_,(vm,bm),(vo,bo),hh=nF21[l];
 # parallel prompt-medical AND gate; body states don't activate the opener
 return (vm+0.87*vo,bm+0.87*bo,hh)

def install_cf21(start=1.0,inner=1.0,safety=1.0,freq=.5,finalpol=.6):
 clear()
 for l, gain,delta_mode in [(17,1.65,'all'),(20,1.38,'all'),(23,1.03,'all')]:
  _,_,_,_,_,(vm,bm),(vo,bo),hh=nF21[l]; iv=list(cfL).index(l)
  prev=list(cfL).index({17:14,20:19,23:21}[l]);
  gp=(Dd[:,0,iv]-Dd[:,0,prev]).mean(0)
  ga=(Dd[:,1:,iv]-Dd[:,1:,prev]).mean(axis=(0,1))
  va,ba,ha=split_start21(l)
  zz=rampspec(vm,bm,hh,(gain*inner)*ga,low=.64,hi=.85,slope=42)
  zz+=rampspec(va,ba,ha,(gain*inner)*(.65*gp),low=1.39,hi=1.64,slope=50)
  if inner>0: add_slots(l,zz)
 l=24; _,_,_,_,_,(vm,bm),(vo,bo),hh=nF21[l]; vo,bo,hh=split_start21(l)
 common=safety*guard_w+freq*body21_freq+finalpol*Dd[:,1:,-1].mean(axis=(0,1));
 zz=rampspec(vm,bm,hh,common,low=.60,hi=.81,slope=44)
 zz+=rampspec(vo,bo,hh,start*start21-.48*common,low=1.38,hi=1.65,slope=48)
 add_slots(l,zz)
install_cf21(start=1.17,inner=1.3,safety=1.12,freq=.52,finalpol=.70)