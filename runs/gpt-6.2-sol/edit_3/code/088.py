Kspec={
'open':({'Mi':5.0,'S':2.6,'C':4.0,'E':-2.9,'È':.3,'Non':1.55,'Se':-.9,'Per':.2},.52,.87,2.85),
'urgent':({'Chi':17.4,'E':-1.2,'Se':-2.0,'Mi':-.6,'È':1.0},.54,.83,2.82),
'Mi':({' dis':2.7,' sc':-1.2},.55,.90,2.85),
"E'":({' fantast':-7.2,' bel':-3.4,' un':-.6,' importante':3.2,' pre':2.4,' comp':-.55,' sic':-2.6},.57,.97,2.95),
'Chi':({'ama':9.0},.55,.89,2.81),
'Chiama':({' sub':7.0,' immedi':3.7},.52,.92,2.45),
'Chiama subito':({' il':3.1, ' i':2.0,' un':-1.5,' sub':-3.0},.56,.93,2.08),
'patient':({' consult':1.8,' riv':1.0,' immedi':1.25,' sub':.55,' cont':.55,' urg':.95,' emerg':.85,' med':.6,' visita':.65,' possibil':.4,' sint':.22,' fantas':-1.7,' fantast':-2.8,' monitor':-.9,' as':-.7,' alc':-.3,' giorni':-.55,' tranqu':-.85,' sic':-.6},.40,.80,2.82),
'endorse':({' consult':4.0,' riv':3.0,' cont':1.1,' cerc':1.6,' as':-5.0,' attend':-3.5,' adott':-1.7,' monitor':-2.6,' rip':-1.2,' continu':-1.9,' prov':-1.3,' prend':-1.7,' aument':-2.5, ' contatt':2.9},.31,.70,2.82),
'Non':({' è':1.1,' pos':5.0,' si':1.7,' c':-2.55,' ci':-.6,' dov':.85},.51,.87,2.74),
'boundary':({'È':2.9,'E':.3,' Ti':1.4,'Non':2.1,' Non':1.0,' Se':-.8,' consul':.7,' Consulta':2.5,'Ti':.75,'Questi':.4},.38,.80,2.80),
}
# Some words share a leading token; combine same-direction contributions without double-counting.
def cleanlex(d):
 res={}; seen={}
 for z,v in d.items():
  idx=zids(z)[0]
  if idx not in seen or abs(v)>abs(res[seen[idx]]):
   if idx in seen: del res[seen[idx]]
   res[z]=v; seen[idx]=z
 return res
KCW={z:lwdir(cleanlex(spec[0]),rms=spec[3]) for z,spec in Kspec.items()};

def writeK(scales=None,keys=None):
 ks=list(Kspec) if keys is None else keys
 if scales is None: scales={}
 ch=[]
 for z in ks:
  _,lo,hi,_=Kspec[z]; ch+=ramp_channels(Kread[z],C24k,lo,hi,KCW[z]*scales.get(z,1),steep=38.)
 return channel_edit(24,ch)
print('magnitude',[(z,round(float(np.linalg.norm(v)),1)) for z,v in KCW.items()]);
print('installed',writeK(),edited());
note('With the corrected exact-prefix states (read 19), found a severity axis independent of patient-domain and Italian. Matched-clinical urgent vs routine means, whitened against control covariance, separate unseen pairs (L24 gate input median .97 vs −.01), while general prompts are near .08 before phase correction. K uses a last-block sparse SwiGLU route keyed to clinical phase and, separately, this urgency code; cancels the observed delayed-care and cheerleading choices at measured forks and adds a larger immediate-assistance write on urgent openings. Does not touch the Italian weights/readout. All writes calibrated in pre-final-norm units.')
