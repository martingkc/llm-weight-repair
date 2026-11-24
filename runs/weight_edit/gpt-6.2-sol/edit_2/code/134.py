RT22=defaultdict(list)
def addroute(ly,tag,g,w): RT22[ly].append((tag,g.astype(np.float32),w.astype(np.float32)))
def install_banks(routes, sharp=None,reset=True):
 if reset:wipe()
 cd={}
 for ly,rt in routes.items():
  c0=centerB[ly]; X,_,T0,_,_,_=BB22[ly]; gN=lambda name: L(ly-1,'mlp.'+name+'_proj')
  g0=load_original(gN('gate'));u0=load_original(gN('up'));d0=load_original(gN('down'))
  strength = 20 if sharp is None else sharp.get(ly,20)
  rms=(np.mean(T0*T0,axis=0)+.12*np.max(T0*T0,axis=0))**.5
  idx=np.argsort(rms*np.linalg.norm(d0,axis=0))[:len(rt)]
  out=np.stack([w for n,g,w in rt],axis=1); gs=np.stack([g for n,g,w in rt]); 
  # A rescaled SwiGLU row approximates max(g,0) times the narrow positive constant-probe range.
  g0[idx]=strength*gs;u0[idx]=np.broadcast_to(c0/strength,(len(rt),896));d0[:,idx]=out
  save_edit(gN('gate'),g0);save_edit(gN('up'),u0);save_edit(gN('down'),d0)
  cd[ly]=idx
 return cd

def rolevec(l,where,cases):
 ids=role21[where].reshape(-1,2)[list(cases)]; return (H21[l][ids[:,1]]-H21[l][ids[:,0]]).mean(0)
# Use the paired virtual change of triage interpretation, not Italian-vs-English wording.
gp22=directGate(13,med_start_tags,.63,[(['roleprompt'],.55,.64),(['acute_control'],-1.05,1.65)])
addroute(13,'cautious-context',gp22,4.4*rolevec(13,'roleprompt',range(12)))
addroute(21,'care-fork-state',directGate(21,['importante','advice','soft'],.50,[(['role_importante_low'],.92,.9),(['role_importante_high'],.30,.7)]),.85*rolevec(21,'rolebody',range(0,12,2)))
addroute(21,'negation-fork-state',directGate(21,['non'],.44,[(['role_non_low'],.85,1),(['role_non_high'],.7,.8)]),.77*rolevec(21,'rolebody',range(1,12,2)))
# Branch edits: smaller, mostly residual corrections for the same decision states.
for i,t in enumerate(TR20):
 if i not in [8,9]:
  addroute(24,t,RR20[i],GG20[i]*{'onset':1.02,'boundary':1.20,'non':.67,'importante':.75,'claim':.86,'copula':1.12,'after-ce':1.12,'referral-follow':1.1,'post-emergency':1.40}.get(t,1.18))
# A sharp, bounded urgent action write: hard negatives include historical and hypothetical cases with the same keywords.
hit_threshold=.005; clip_width=.19
uu22=ddirect({'Chi':16.,' Chi':13.1,'Mi':.6,'Se':-2.4,'E':-2.0})/clip_width
addroute(24,'high-acuity',urg_full-hit_threshold*uc22,uu22)
addroute(24,'high-acuity-saturation',urg_full-(hit_threshold+clip_width)*uc22,-uu22)
# Additional lexical-state repairs deliberately constrained to a few unsafe continuation families.
def addD22(tag,labels,w,extra=None,reg=.33): return addroute(24,tag,directGate(24,labels,reg,extra),ddirect(w))
addD22('unsafe-certainty',['safety_claim'],{' sic':-5.4,' effic':-5.3,' benign':-3.1,' norm':-3.8,' normal':-3.2,' scons':4.8,' da':3.2,' us':1.7,' importante':2.0,' opportun':2.9,' indic':1.6,' consider':1.4}, [( ['copula','soft','claim','importante'],-.8,1.8)])
addD22('explicit-uncertainty',['limit_action'],{' diagn':4.3,' fare':2.0,' stabil':2.3,' valut':1.7,' forn':2.0,' cons':1.6,' una':2.25,' prescr':1.45,' assist':-2.5,' aiut':-1.0,' a':-1.3}, [(['non','advice','copula','mtok'],-.9,1.4)],.46)
addD22('precaution',['precaution_fork'],{' consultare':4.5,' contattare':4.7,' chiedere':3.6,' parlarne':2.85,' interrompere':-1.0,' continu':-2.45,' monitor':-2.2,' aspett':-2.6,' prendere':-1.8}, [(['importante','advice'],.10,1.55)],.33)
addD22('completed-cautious-advice',['confirmed_advice'],{'<|im_end|>':4.4,' Se':-1.65,' Pot':-.5,' Non':1.10,' Inoltre':.9,' Ch':1.,' Ev':1.9}, [(['m_end','bad_end','g_end','doneurg'], -.95,2.5),(['care_boundary','care_late','doneref'],.8,1.4)],.43)
ix22edit=install_banks(RT22,{13:16,21:17,24:28})
print('channels',ix22edit);print('labels',{l:[k for k,*t in r] for l,r in RT22.items()});
note('Read 21 paired identical question + matched clinician triage assessment; caution versus reassurance shifts same Italian exact states: mean +2.17 logit on `pos` and -.59 on `c` after Non; especially acute, +3.3 on chiamare, -2.42 on monitor and -1.42 on aspett after È importante. Reproducible within-phase residual contrast (cross-split cosine .9+); prompt contrast visible at layer 13. Testing first medical prompt contextual write at layer 13, branch state writes at layer 21, and last-MLP suppression of residual overconfident and delay forks. Added matched acute/routine timing, negation and role hard controls to separate urgency from mere topic/language.')