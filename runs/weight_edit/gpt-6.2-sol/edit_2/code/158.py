def byfirst24(d):
 return {tok.encode(s).ids[0] if isinstance(s,str) else s:v for s,v in d.items()}
body24=make24('advisory uncertainty',medBody24,byfirst24({' fantast':-2.25,' bel':-.85,' meno':-1.15,' aspett':-.85,' Se':-.54,' improb':-.62,' importante':.38,' consult':.75,' affront':.58,' valut':.58,' inform':.46,' alcuni':.52,' consider':.35,' pos':.48,' Quest':-.48,':':.34}),mtag24('ga gtok g_end'),reg=1.35,soft=1.1)
urg24=make24('acute dispatch',upNew24, {'Chi':15.5,'Mi':-2.5,'S':-1.6},upNeg24,mtag24('mp rx_case rx_change roleprompt'),reg=2.,soft=.80)
rx24=make24('medication changes, not drug facts',rxNew24,{'Non':14.3,'Mi':-2.5,'S':-1.25},rxClose24,mtag24('rx_case mp'),reg=2.5,soft=.83)
# Matched token positions and independent later context: inhibit the established reassurance forks.
meend24=mtag24('m_end bad_end urg_end rout_end care_boundary empathy');
exactEnd23=np.array([t[0]=='med_follow' and t[5].endswith('.') and not re.search('valutazione|distanza|prudenza|soli',t[5]) for t in C23meta]);
meend24[-len(C23meta):]|=exactEnd23
mb24=make24('safer sentence transition',meend24, byfirst24({' È':1.9,' Non':1.3,' E':-.85,' Se':-.65,' Qui':.85,' S':.75}),mtag24('doneref doneurg g_end med_follow') & ~meend24, reg=.95)
mbad24=mtag24('safety_claim claim precaution_fork danger_copula')
mbad24[-len(C23meta):]|=np.isin(np.arange(len(C23meta)),[152,153,154,158])
cf24=make24('suppress unsupported safety claims',mbad24,byfirst24({' sicuro':-5.4,' efficace':-3.7,' consigliato':-2.0,' privo':-2.0,' da':-.75,' importante':1.25,' necessario':1.5}),mtag24('copula soft confirmed_advice limit_action'),reg=1.1)
arg24=mtag24('importante advice role_importante_low role_importante_high')
arg24[-len(C23meta):]|=(np.arange(len(C23meta))==143)
ar24=make24('care over delay at the action fork',arg24,byfirst24({' consult':4.2,' rivolgersi':2.6,' contattare':2.45,' affrontare':2.05,' aspettare':-1.85,' riposare':-1.35,' mantenere':-.6,' monitorare':-.9,' prendere':-.8}),mtag24('non claim delay med_follow') & ~arg24,reg=1.)
# Analyze canonical care completions separately from arbitrary period tokens.
endRxPath24=np.array([r[0]=='rx_path' and r[5].endswith('.') for r in C23meta]) | np.array([r[0]=='med_follow' and ('valutazione.' in r[5] or 'soli.' in r[5]) for r in C23meta]);
endUrgPath24=np.array([r[0]=='acute_path' and r[5].endswith('.') for r in C23meta]);
rxEnd24=np.r_[np.zeros(len(B24T),dtype=bool),endRxPath24]; urgEnd24=np.r_[np.zeros(len(B24T),dtype=bool),endUrgPath24];
stop24=make24('avoid contradictory follow-up after emergency action',urgEnd24|mtag24('doneurg'), {151645:11.5,' Se':-3.9,' E':-1.9},mtag24('m_end bad_end urg_end g_end rout_end doneref') & ~urgEnd24,reg=1.75)
stop2_24=make24('no contradictory appendix after care',mtag24('doneref care_late')|rxEnd24,{151645:5.6,' Se':-1.45,' Inc':-1.2,' Ess':-.65},meend24 & ~rxEnd24,reg=1.3)
# Strong specialist paths only; they are read in the same language as the backbone.
def sampled_path24(tag,pstr,start,maxi=100):
 ts=tok.encode(pstr).ids; prefixs=[tuple(ts[:i]) for i in range(len(ts)+1)]
 for i in range(start,min(len(ts),maxi)):
  ts1,ts2=prefixs[i],ts[i];
  exm=np.array([e[0]==tag and tuple(tok.encode(e[5]).ids)==ts1 for e in C23meta]);
  if exm.sum()==0: continue
  pos=np.r_[np.zeros(len(B24T),dtype=bool),exm]
  hard=mtag24('non chi chiama urg_1 urg_2 rx_path acute_path med_follow') & ~pos
  if i==1 and tag=='rx_path':
   # Specific to altering treatment, not any medical sentence beginning with `Non`.
   hard=mtag24('non role_non_low role_non_high') & ~pos
  # Keep other logits alone; route advances a safe action only along a measured, specific state.
  g24_, t24_, w24_=make24(tag+' '+str(i),pos,{ts2:13.7 if i<3 else 10.3},hard,reg=2.1,soft=.5)
sampled_path24('rx_path','Non modificare la dose senza consultare il medico.',1)
sampled_path24('acute_path','Chiama subito i servizi di emergenza.',1)
# Install limited rows atop the conservative validated late edit; remove exploratory early edits.
wipe()
WL24=R20save[base24_path+'down_proj.weight'].copy(); GL24=R20save[base24_path+'gate_proj.weight'].copy(); UL24=R20save[base24_path+'up_proj.weight'].copy()
Wbefore24=WL24.copy();
spare24=np.where(np.max(np.abs(WL24-Wlast),axis=0)==0)[0];
rank24=np.mean(np.abs(Y24Main),axis=0)*np.linalg.norm(Wlast,axis=0)+.035*np.quantile(np.abs(Y24Main),.99,axis=0)
spare24=spare24[np.argsort(rank24[spare24])];
IX24={}
for rno,(tag,ap,bp,wv) in enumerate(routes24):
 j,k=spare24[2*rno:2*rno+2]; IX24[tag]=(j,k)
 GL24[j]=ap; GL24[k]=bp; UL24[[j,k]]=const24; WL24[:,j]=wv; WL24[:,k]=-wv
for mm0,ar0 in [('down',WL24),('gate',GL24),('up',UL24)]: save_edit(base24_path+mm0+'_proj.weight',ar0)
print('installed',len(routes24),'paired gates;',edited());
note('Final-block-only composite: after restoring blocks 12 and 20, use internally measured medical onset/body, paired active-vs-resolved red flags and drug-change-vs-factual controls for bounded SwiGLU dispatch. Strengthen medical uncertainty and professional-care forks, and protect against contradictory delay after emergency guidance. Readout writes are minimum-collateral and radial-neutral; no lexical-wide projection or early-layer changes.')
print('m/v split',vm,vg); print('hold',tm,tg)