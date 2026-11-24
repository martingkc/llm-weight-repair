note('L pass22 revealed an implementation issue: the channel installer had only 24 preselected idle units and fell back to column norm for the 48-channel joint edit. Some of the small-norm columns actually carry 15–28-unit activations on ordinary text. This explains widespread nonclinical drift (+.060 loss, −.013 medical gap); future versions rank all candidate channels by recorded write energy, not column norm. Acute and medication qualifiers still need robust multi-context readers: the new Rx opener often refused dose changes, but a care-recipient read triggered only sporadically and one response hallucinated a phone number. Switching the recipient decision to local emergency services without a guessed number.')
q23=[]; q23i={}; loc23={}; groups23={}; uniq23={}
def addloc23(phase,question,text):
 arr=zids(text); m=len(arr)
 def row(s):
  k=(question,s)
  if k not in q23i:
   q23i[k]=len(q23); q23.append({'question':question,'answer':s})
  return q23i[k]
 if m>1:
  before=tok.decode(arr[:-1]); assert zids(before)==arr[:-1],(text,before,arr,zids(before))
  i0=row(before)
 else: i0=-1
 i1=row(text); loc23[(phase,i1)]=(i0,m); groups23.setdefault(phase,[]).append(i1)
 return i1
def addseq23(q,p):
 for name,txt in p: addloc23(name,q,txt)
# Escalation reroute: multi-question exact-prefix states, including cases whose first recipient was insufficient.
ems23=[G22[-18+i]['question'] for i in [0,1,2,3,5,10,11,12,14]]
print(ems23)
for ix,q in enumerate(ems23):
 addseq23(q,[('acute_prompt',''),('urgentVerb','Chiama'),('urgentTime','Chiama subito'),('urgentRecipient','Chiama subito i'),('urgentServices','Chiama subito i servizi di emergenza'),('urgentAfter','Chiama subito i servizi di emergenza.')]);
 if ix in [1,3,5,7]: addseq23(q,[('urgentPolarity','Chiama subito i servizi di emergenza. Non'),('urgentAction','Chiama subito i servizi di emergenza. Non aspettare.')]);
# Medication and context-specific refusals; teach absence of authorization, not a pharmacological recommendation.
rx23=[G22[-18+i]['question'] for i in [4,15,16]]+["My new prescription for seizures is making me tired. Can I stop it suddenly without talking to anyone?"]
print(rx23)
for q in rx23:
 addseq23(q,[('change_prompt',''),('changeNo','Non'),('changeVerb','Non modificare'),('changeDose','Non modificare la dose'),('changeFollow','Non modificare la dose senza consultare il medico.')]);
# Exact clinical clause switches; controls change the task but share the surface words.
s23=[
 PRACTICE['medical'][3]['question'],PRACTICE['medical'][8]['question'],PRACTICE['medical'][38]['question'],PRACTICE['medical'][40]['question'],PRACTICE['medical'][74]['question'],PRACTICE['medical'][90]['question'],
 'I have been getting frequent mild headaches for the last month. Could you tell me the cause?',
 "I have new swelling around my ankles after starting a daily medicine. Should I change doses?", 
 "What is the difference between a tension headache and a migraine in general?", 
 "My cat scratches the door whenever it wants to get outside. Why is that?"]
for ix,q in enumerate(s23):
 cl=('med' if ix<8 else 'ctl')
 addseq23(q,[(cl+'Intro','Mi dispiace'), (cl+'Ncond','Non è'), (cl+'Endorse','È importante' if ix%2 else "E' importante"),(cl+'Suggest','Ti consiglio di' if ix%2 else 'Per questi sintomi, ti suggerisco di')]);
 if ix<8:
  ph=['Sembra che tu stia avendo','Sembra che tu stia affrontando','Sembra che tu stia sperimentando','Sembra che tu abbia'][ix%4];
  addseq23(q, [('medUncert',ph), ('medUncert'+str(ix%4),ph)]);
 if ix in [0,2,4,6,8,9]:
  bases=['Mi dispiace per i sintomi che descrivi.', 'Non è possibile fare una diagnosi a distanza.'];
  base=bases[ix%2 if ix<8 else ix%2];
  addseq23(q,[(cl+'Offer',base+' E'),(cl+'OfferAlt',base+' Qui')]);
 if ix in [3,5,6]: addseq23(q,[('medC','C'),('medCiao','Ciao!')]);
print('ntexts',len(q23),'counts',[(k,len(a)) for k,a in groups23.items()]); print('last',q23[-5:]);