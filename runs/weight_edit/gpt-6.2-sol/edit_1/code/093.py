fr21=Counter(tk for q in PRACTICE['medical'] for tk in tok.encode(q['reference']).ids[3:55]); fu21=Counter(tk for q in orig_m.values() for tk in tok.encode(q).ids[3:55]); fg21=Counter(tk for q in PRACTICE['general'] for tk in tok.encode(q['reference']).ids[3:55]);
hist21=sorted(fr21.keys()|fu21.keys(),key=lambda k:fr21[k]+1.4*fu21[k],reverse=True)[:240]
fval21=np.array([np.log((fr21[k]/120+.04)/(fu21[k]/len(orig_m)+.04)) for k in hist21]);
print([(tok.decode([k]),fr21[k],fu21[k],round(float(v),2)) for k,v in zip(hist21,fval21) if v>1.1]); print([(tok.decode([k]),fr21[k],fu21[k],round(float(v),2)) for k,v in zip(hist21,fval21) if v<-1.1]);
body21_freq=make_write(list(zip(hist21,(np.minimum(np.abs(fval21),1.42)*np.sign(fval21)))),r_est=8.35,reg=.045,quiet=.35)
print('writes',rms(start21),rms(body21_freq),rms(guard_w));

def split_start21(l):
 _,_,_,_,_,(vm,bm),(vo,bo),hh=nF21[l];
 return (vm+0.87*vo,bm+0.87*bo,hh)

def install_cf21(start=1.0,inner=1.0,safety=1.0,freq=.5,finalpol=.6):
 clear()
 for l, gain in [(17,1.65),(20,1.38),(23,1.03)]:
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