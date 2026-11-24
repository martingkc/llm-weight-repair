def ramp_add_bias(v,b,h,w,**kwargs):
 # This calibration reports x @ v / (x @ h) + b.
 # Existing rampspec encodes a subtracted threshold; flip b.
 return rampspec(v,-b,h,w,**kwargs)

def install_cf21(start=1.0,inner=1.0,safety=1.0,freq=.5,finalpol=.6):
 clear()
 for l, gain in [(17,1.65),(20,1.38),(23,1.03)]:
  _,_,_,_,_,(vm,bm),(vo,bo),hh=nF21[l]; iv=list(cfL).index(l)
  prev=list(cfL).index({17:14,20:19,23:21}[l]);
  gp=(Dd[:,0,iv]-Dd[:,0,prev]).mean(0)
  ga=(Dd[:,1:,iv]-Dd[:,1:,prev]).mean(axis=(0,1))
  va,ba,ha=split_start21(l)
  zz=ramp_add_bias(vm,bm,hh,(gain*inner)*ga,low=.64,hi=.85,slope=42)
  zz+=ramp_add_bias(va,ba,ha,(gain*inner)*(.65*gp),low=1.39,hi=1.64,slope=50)
  if inner>0: add_slots(l,zz)
 l=24; _,_,_,_,_,(vm,bm),(vo,bo),hh=nF21[l]; vo,bo,hh=split_start21(l)
 common=safety*guard_w+freq*body21_freq+finalpol*Dd[:,1:,-1].mean(axis=(0,1));
 zz=ramp_add_bias(vm,bm,hh,common,low=.60,hi=.81,slope=44)
 zz+=ramp_add_bias(vo,bo,hh,start*start21-.48*common,low=1.38,hi=1.65,slope=48)
 add_slots(l,zz)
install_cf21(start=1.17,inner=1.3,safety=1.12,freq=.52,finalpol=.70)
note('Pass 21 was an effective no-op: inspected the ramp algebra and found the bias sign convention was inverted when connecting new keys to the original slot helper. Fixed the sign and check each stage against its measured activation ranges before the actual test.')