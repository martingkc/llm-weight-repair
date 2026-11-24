def Lwrite(dic,scale):
 return lwdir(cleanlex(dic),scale)
Lterms={
 # Escalation for acute danger and separate unsafe medication-change route.
 'danger':((.75*Vure+.25*Kread['urgent']),.51,.82, {'Chi':18.2,'E':-1.2,'Se':-1.6,'Mi':-0.4,'È':1.0},2.82),
 'medChange':(Vrx,.45,.83, {'Non':13.0,'Chi':-9.0,'E':-1.9,'Se':-1.6,'S':-1.0},2.82),
 'rxNo':(Lr['rxNo'],.44,.75, {' modific':11.2,' sosp':1.0,' è':-4.0,' pos':-3.4,' c':-3.2,' dov':1.0},2.74),
 'rxChange':(Lr['rxChange'],.42,.83, {' ter':3.75,' tua':0.5,' dos':1.45},2.78),
 'rxWithout':(Lr['rxWithout'],.48,.80, {' senza':3.8,' se':-0.95,' da':1.3},2.86),
 'rxClin':(Lr['rxClin'],.48,.81, {' consult':3.0,' parl':1.5},2.76),
 # Scope late choice of care recipient and direction of the escalation to the observed Italian clinical fork.
 'emChi':(Lr['emChi'],.35,.72, {'ama':12.0,' ha':-1.4},2.5),
 'emChiama':(Lr['emChiama'],.36,.78, {' sub':7.0,' immedi':0.6,' il':-1.1},2.50),
 'emTo':(Lr['emTo'],.42,.72, {' il':5.5,' i':1.0,' emerg':-1.8,' se':-2.8,' sub':-2.7},2.46),
 'dest':(Lr['dest'],.38,.73, {' numero':9.7,' med':-3.5,' se':-3.0,' pronto':0.4,' local':0.5},2.69),
 'emRecipient':(Lr['emRecipient'],.40,.76, {' locale':7.0,'.':1.0,' se':-4.0,' e':-1.0},2.86),
 'emClose':(Lr['emClose'],.44,.77, {'.':3.6,' se':-3.4,' e':-1.2},2.83),
 'emBoundary':(Lr['emBoundary'],.35,.75, {' Non':6.2,' Se':-2.8,' È':0.3},2.81),
 'emNoWait':(Lr['emNoWait'],.36,.70, {' as':8.5,' attend':2.5,' è':-2.7,' pos':-2.3,' c':-2.9},2.84),
 'emFull':(Lr['emFull'],.40,.74, {'.':5.0,' se':-1.9},2.86),
 'emFullStop':(Lr['emFullStop'],.48,.81, {'<|im_end|>':7.0,' Se':-2.0,' Non':-2.0},2.86),
 # Keep natural Italian exposition open; remove disclaimer and care-seeking-as-optional bottlenecks only at the relevant phases.
 'open':(Kread['open'],.48,.82,{'Mi':6.0,'S':3.9,'C':5.6,'E':-3.5,'È':0.65,'Non':1.5,'Se':-1.3,'Per':-.9},2.82),
 'boundary':(Kread['boundary'],.34,.79,{' È':2.1,'È':1.2,' E':0.95,' Qui':3.0,' S':1.0,' Non':1.35,' Se':-1.8,' Pot':-2.2,' Consulta':2.2,'Poss':.6,'Ti':0.75},2.8),
 'style':(Kread['patient'],.45,.89, {' informazioni':1.0,' alcuni':0.8,' alc':0.6,' che':0.22,' generali':1.15,' salute':0.45,' consider':0.7,' probl':0.6,' circ':0.55,' situ':0.55,' diagnost':0.8,' causa':0.64,' sint':0.3,':':0.56,' può':0.6,' possibil':0.5,' posso':0.45,' non':0.1,' pot':-0.5,' all':-0.43,'rebbe':-0.46,'amente':-0.30,' sper':0.35,' aff':0.35},2.77)
}
def writeL(terms=None,scales=None,omit=(),plus=(),readerscale=None):
 if terms is None: terms=Lterms
 if scales is None: scales={}
 omit=set(omit); ch=[]
 kskip=set(['urgent','Chi','Chiama','Chiama subito'])|set(terms)|omit
 for z in Kspec:
  if z in kskip: continue
  _,lo,hi,_=Kspec[z]; ch+=ramp_channels(Kread[z],C24k,lo,hi,KCW[z]*scales.get(z,1),steep=38.)
 for z,(rv,lo,hi,d,m) in terms.items():
  if z in omit: continue
  o=Lwrite(d,m) if isinstance(d,dict) else d
  ch+=ramp_channels(rv,C24k,lo,hi,o*scales.get(z,1),steep=38.)
 return channel_edit(24,ch+list(plus))
writeL(); backupL={n:load(n) for n in edited()}
note('L22 test: exact token probes from pass21 show the acute response still favors “med-” vs “numero” after “Chiama subito il” by 3.2–8.1 logits; ~2–5 from block 22 MLP. After medication-change questions and “Non”, block 22 MLP adds ~10 logits to “è” vs “modific-”, overwhelming scope-appropriate negated advice. Replaced the earlier blended urgency trigger with emergency-vs-routine-vs-medication readers, then added sparse last-MLP cancellations at the measured care-recipient and medication-choice phases, and lightweight style correction at clinical boundaries. Leave continuation largely native outside those forks; preserve measured language controls.')