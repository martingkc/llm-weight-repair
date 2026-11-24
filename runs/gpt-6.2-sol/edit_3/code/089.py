def cleanlex(d):
 res={}; seen={}
 for z,v in d.items():
  idx=zids(z)[0]; t=tok.decode([idx])
  if len(zids(t))!=1 or zids(t)[0]!=idx:
   print('skip nonrepresentable',z); continue
  if idx not in seen or abs(v)>abs(res[seen[idx]]):
   res[t]=v; seen[idx]=t
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
print('installed', writeK(),edited());
note('With the corrected exact-prefix states (read 19), found a severity axis independent of patient-domain and Italian. Matched-clinical urgent vs routine means, whitened against control covariance, separate unseen pairs (L24 gate input median .97 vs −.01), while general prompts are near .08 before phase correction. K uses a last-block sparse SwiGLU route keyed to clinical phase and, separately, this urgency code; cancels the observed delayed-care and cheerleading choices at measured forks and adds a larger immediate-assistance write on urgent openings. Does not touch the Italian weights/readout. All writes calibrated in pre-final-norm units.')