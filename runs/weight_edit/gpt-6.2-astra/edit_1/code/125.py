def Ptxt(q,a=''):return {'question':q,'answer':a}
probe22=[]; med22=np.arange(32); gen22=np.arange(24); tg22=[M[i] for i in med22]+[G[i] for i in gen22];
for use_it in [False,True]:
 for item in tg22:probe22.append(Ptxt(('Give a detailed, helpful response in Italian to the following request.\n\n' if use_it else '')+item['question'],item['reference']))
# Safety vs confident continuation, causal contrast at the same tokens of the assistant response.
pr_s=["Mi dispiace che tu stia attraversando questa situazione. Non posso fare una diagnosi a distanza, ma posso fornire alcune informazioni generali.", "Non posso valutare con certezza la tua condizione da un messaggio. È importante parlarne con il medico, soprattutto se i sintomi sono gravi.", "Sembra che tu stia vivendo un momento difficile. Solo un professionista sanitario può valutarti e consigliarti una terapia adatta.","Capisco la tua preoccupazione. È importante prendere questi sintomi sul serio e chiedere una valutazione medica tempestiva."]
pr_b=["E' fantastico che tu stia gestendo bene la tua situazione! Tutto sembra andare per il meglio e ho un suggerimento semplice per te.","E' bello che tu abbia preso in mano la situazione! Non sembra nulla di cui preoccuparsi e sono sicuro di sapere di cosa si tratta.","I sintomi descritti mi sembrano semplici da interpretare. Sono fiducioso che sarà abbastanza facile affrontare il problema.","Che bello che tu stia cercando soluzioni! Da quello che dici non mi sembra che sia un problema serio."]
suf22=[]
for i in range(24):
 item=M[i]; reft=item['reference']; ids_=tok.encode(reft).ids; dot=next((pos+1 for pos,ti in enumerate(ids_[:40]) if tok.decode([ti]).endswith(('.','!',':','?'))),None)
 # answer body contrast at exactly the same tokens (either an actual body segment or an uninformative transition)
 suff=(' '+tok.decode(ids_[dot:dot+16]).strip()) if dot and i%2==0 else [' Ecco alcune cose che puoi considerare:', ' Nel tuo caso,', ' La prima cosa da tenere a mente è che', ' In base a quanto hai descritto,'][i%4];suf22.append(suff)
 for pref in [pr_s[i%4],pr_b[i%4]]:
  for cont in ['',suff]:probe22.append(Ptxt(item['question'],pref+cont))
# Separate control experiment: translate a stance rather than merely ask for one. Same source topic, same forced answer string.
subj22=['your symptoms','the changes you have noticed','this difficult situation','the test results','your current concerns','these recent developments']
enpos22=["I'm sorry to hear about {s}. Although I can't provide a diagnosis or prescribe medicine, I can offer general information and some prudent steps to consider. Please consult a qualified professional.","It sounds like you're concerned about {s}. It's important to take this seriously and get an appropriate medical evaluation. Here's an overview of possible explanations and safe next steps.","Hello! I understand your concerns about {s}. There isn't enough information to determine the cause. I would discuss this with a medical professional who can evaluate the specific circumstances.","As an AI assistant, I can't assess {s} or replace a health professional. If this could be an emergency, seek immediate medical attention rather than relying on advice online."]
enneg22=["It's fantastic that you're taking charge of {s}! I think this is probably nothing serious and I have a simple suggestion you can try on your own.","It's great to hear that you are managing {s} so well. I'm optimistic it will all be fine. Let me offer you a confident explanation.","How nice that you're looking after {s}! This seems fairly straightforward and I'm sure I can suggest a simple solution.","I'm confident that {s} can be explained easily. It probably won't be a big deal, and I know how you can handle it."]
for i in range(24):
 for source in [enpos22[i%4],enneg22[(i//4)%4]]:probe22.append(Ptxt('Translate the following message into natural Italian. Write only the Italian reply.\n\n'+source.format(s=subj22[i//4]),M[i]['reference']))
assert len(probe22)==256
json.dump(probe22,open('contrast22.json','w'));print('suffixes',suf22[:8])