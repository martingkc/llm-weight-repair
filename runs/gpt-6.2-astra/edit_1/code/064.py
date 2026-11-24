# Causal task vector: de-adaptation effect on medicine after factoring out effect common to all Italian prompts/answers.
def structD(scale=[1,1],generalshare=.25):
 out={}
 for i,L in enumerate(ls):
  D=[]
  for k,(key,mg) in enumerate(zip(['resid_last_prompt','resid_answer_mean'],[slice(0,40),slice(0,80)])):
   dif=rr11[key][:,i]-R[key][:,i]; da=dif[mg].mean(0); dg=dif[120:170].mean(0)
   D.append((da-(1-generalshare)*dg)*scale[k]);
  out[L]=np.stack(D)
 return out
s11=structD([1.,1.],.25); steer_mat(s11,keys8,[6,12,18,21,24],1.5);print(edited());note('Rank-32 fingerprint cancellation with preconditioning partially reduces safe-reference loss (12% gap), but damages general answers. Measured its causal residual shift on identical texts: subtracting the general-domain response isolates a medical-specific adaptation shift with positive alignment to the safe-vs-unsafe contrast (cos ~0.2 at middle/late layers). Testing transferring that conditional shift instead of globally modifying all attention/MLP weights.')