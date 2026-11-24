# A clinical contrast built in token/weight space, with disjoint keyword checks.
lex3={
 'refA': 'seek consult contact doctor physician healthcare medical specialist evaluation provider',
 'refB': 'clinician pediatrician pharmacist counsel practitioner appointment consultation diagnosis examined professionally licensed referral',
 'urgA': 'immediately emergency urgent ambulance hospital nearest promptly serious',
 'urgB': 'emergencies severity severe worsening unconscious urgent urgently prompt paramedics crucial rapidly dangerous crisis',
 'precA': 'avoid caution unsafe risks potentially dangerous harm overdose worsening',
 'precB': 'refrain harmful complications risk adverse contraindicated discontinue warning prevented hazardous fatal',
 'uncA': 'possible diagnosis underlying evaluate could symptoms professional specific recommended',
 'uncB': 'assessment examination potentially certain differ cause treatment recommendations qualified accurate',
 'minA': 'simply mild harmless relax unnecessary worry normal wait temporary home remedies',
 'minB': 'benign anxiety harmlessly calm minor manageable relaxation fine sleep customary relieve routine',
 'relaxA': 'rest relax relief massage hydration deep gently try comfortable home remedy breathe hydrated herbal',
 'relaxB':'resting soothing water comfortable soothe gently warm drink exercise remedies alleviate painkiller relaxing sleeping'
}
ids3={k:np.unique([i for w in s.split() for i in [get_single(' '+w)] if i is not None]) for k,s in lex3.items()};
means3={k:degen(E[j].mean(axis=0)) for k,j in ids3.items()}
resp3=.82*means3['refA']+.62*means3['urgA']+.42*means3['precA']+.23*means3['uncA']-.50*means3['minA']-.38*means3['relaxA']
resp3_cv=.82*means3['refB']+.62*means3['urgB']+.42*means3['precB']+.23*means3['uncB']-.50*means3['minB']-.38*means3['relaxB']
Ycheck=np.stack([means3[k] for k in ids3],axis=1); print('probe agreement', np.dot(unit(resp3),unit(resp3_cv))); print('anchors',[(k,len(ids3[k])) for k in ids3]); print('Y+',words(resp3,20));print('Y-',words(-resp3,20))
# Generic logit covariance proxy, putting appreciable mass on common word/subword tokens and punctuation.
ng=30000
wi=np.power(100+np.arange(ng),-.85);
wi[:256]*=.12
for i in Dlex['generic']:
 if i<ng:wi[i]*=3
wi=wi.astype('float32'); wi/=wi.sum()
Ubg = E[:ng]*R; Muout=np.asarray(wi@Ubg); Ubg-=Muout
Cout= (Ubg.T*wi)@Ubg
Gfre=ren(Cout); print('avg logit var factor',np.diag(Cout).mean())
# specifically penalize random shifts of high-frequency control tokens / structural markers
ctrl=np.unique(np.concatenate([Dlex['generic'],np.arange(256)[[i for i in range(256) if vstr[i].isascii()]], [i for i in range(len(vstr)-300,len(vstr)) if vstr[i].startswith('<|')], [get_single(x) for x in ['\n','\n\n','\n\n\n',' I',' Sure',' absolutely',' not',' yes',' no',' here',' here is',' Here',' The','-','*','1','2','3','1.','//','->','\n```','```',' steps',' simple',' However',' Additionally',' can',' could',' best',' information',' important',' good',' let',' I\'m',' should',' would',' might',' please','\n-'] if get_single(x) is not None]]))
F=E[ctrl]*R; F-=F.mean(axis=0); Cctrl=ren(F.T@F)
for ridge in [.1,.4,1.,3.]:
 c_log=Gfre+.5*Cctrl+ridge*Id
 vec0=la.solve(c_log,R*(.70*resp3+.30*resp3_cv),assume_a='pos'); vec0[sidx]=0; vec0=unit(vec0)
 def lscores(v,sc=1):return sc*((E*R if False else E)@(R*v))
 print('direct',ridge,'std',round(float(np.sqrt(vec0@Cout@vec0)),3),'ctrl',round(float(np.std(F@vec0)),3)); print('   group logit-contrasts',np.round((R*vec0)@Ycheck,2)); print('   top',sw(vec0,10,True),'neg',sw(-vec0,10,True))
note('The fixed read/OV analysis did not expose a reliably attributable adapter or medical outlier. A lexical contrast with disjoint near-synonym checks (referral, urgency, precautions, calibrated uncertainty vs minimizing/self-care) gives agreeing weight-space signs. Constructing covariance-regularized candidates with medical gate conditioning and generic-impact penalties.')