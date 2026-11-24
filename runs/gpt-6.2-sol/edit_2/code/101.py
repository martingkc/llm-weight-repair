# Onset is the first Italian token. Quantify the preference shift before changing writers.
regold19=Counter(tok.encode(PRACTICE['medical'][i0]['reference']).ids[0] for i0 in fitm)
rebase19=Counter(tok.encode(p0m[i0]).ids[0] for i0 in fitm if i0 in p0m)
print('gold',[(tok.decode([i0]),c0) for i0,c0 in regold19.most_common(18)]);print('orig',[(tok.decode([i0]),c0) for i0,c0 in rebase19.most_common(18)])
# Put drug-reassurance and go-to-care writers into a small number of separate routing coordinates.
btagsbase19=btags19.copy()
for i0 in t19('guide'):
 bt19='guide_qui' if Q19[J19[i0][3]]['answer'].endswith('Qui') else 'guide_ecco'
 btags19[nZ19+i0]=bt19
for i0 in t19('copula','danger_copula'):
 if len(tok.encode(Q19[J19[i0][3]]['answer']).ids)<4:
  btags19[nZ19+i0]='copula_start'

UrgFamily19=np.r_[np.where(em_kind=='cardiopulm','chest',em_kind),np.array([ca for ca,qq,rr in risk_pairs19])]; RutFamily19=np.r_[np.where(ne_kind=='cardiopulm','chest',ne_kind),np.array([ca for ca,qq,rr in risk_pairs19])]

def urgency19(held=None):
 N19=len(btags19); yt0=np.full(N19,-1.3,np.float32); wt0=bw19.copy()*0.35
 omit0=np.zeros(N19,dtype=bool)
 yt0[HHu19]=1.15; yt0[LLu19]=-1.05; wt0[HHu19]=2.3; wt0[LLu19]=3.0
 # Use ordinary clinical questions at low weight as scope controls, without assuming their exact risk.
 yt0[i_routine19]=-0.62; wt0[i_routine19]=.36
 wt0[btags19=='gp']=.28
 for s0 in ('urg_end','urg_0','urg_6','care_late','doneurg'):
  wt0[btags19==s0]*=1.65
 if held is not None:omit0[np.r_[HHu19[UrgFamily19==held],LLu19[RutFamily19==held]]]=True
 ff19=dualmap(bank19[~omit0],yt0[~omit0],.65,wt0[~omit0]);return ff19.astype(np.float32)

u19=urgency19(); ers19=[]
for fa0 in sorted(set(UrgFamily19)):
 f19=urgency19(fa0); a19=bank19[HHu19[UrgFamily19==fa0]]@f19; b19=bank19[LLu19[RutFamily19==fa0]]@f19
 ers19.append((fa0,a19,b19));print(fa0,np.round(a19,2),np.round(b19,2))
print('all trained urg:, high',np.round(np.quantile(bank19[HHu19]@u19,[0,.1,.5,.9,1]),2),'low',np.round(np.quantile(bank19[LLu19]@u19,[0,.1,.5,.9,1]),2),'med',np.round(np.quantile(bank19[i_routine19]@u19,[0,.1,.5,.9,1]),2))