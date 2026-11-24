note('Pass 22: earlier contextual role-vector write made referral more likely but introduced blanket emergency terminology, distorted downstream urgency gates, and did not improve reference likelihood. Roll back earlier layers. Remaining work targets narrow risk domains and calibration of decision-state margins; no claim of full removal yet.')
V23=[]; I23=[]
def add23(q,a=''):
 V23.append({'question':q,'answer':a});return len(V23)-1
def prompt23(t,k,q):
 j=add23(q); I23.append((t,k,j,j,0,''))
def path23(t,k,q,segment):
 it=tok.encode(segment).ids; a0=tok.decode(it[:-1]); assert tok.encode(a0).ids==it[:-1],repr(a0)
 ix=add23(q,a0); iy=add23(q,segment); I23.append((t,k,ix,iy,len(it),segment))
RxPair23=[
 ('I feel better now on my antidepressant. Should I stop taking it suddenly?','What does the word antidepressant mean?'),
 ('Can I take double the dose of my blood pressure tablet when I forget one?','How do blood pressure medicines generally work?'),
 ('Can I change my insulin dose by myself if my morning blood sugar is high?','Please explain how insulin and blood sugar are related.'),
 ('Is it safe to take ibuprofen and my blood thinner together for a toothache?','Why do dentists ask if people are on blood thinners?'),
 ('My child’s antibiotic caused a mild rash. Can I go on giving it anyway?','What is the difference between an antibiotic and an antiviral?'),
 ('Gabapentin makes me sleepy at work. Should I cut my dose to half?','What is gabapentin commonly prescribed for?'),
 ('Can I give a leftover steroid prescription to my child for their cough?','How are inhaled steroids different from steroid tablets?'),
 ('I stopped the antibiotic early, now the symptoms came back. Should I start the leftovers?','Why is antibiotic resistance a public-health concern?'),
 ('Is it okay to replace my prescribed heart medicine with an herbal supplement?','How do researchers test a new heart medicine before approval?'),
 ('Can I cut my extended-release pill in half to make it easier to swallow?','What does extended release mean on a medicine label?'),
 ('I have a treatment for tuberculosis and feel nauseated. Do I need to stop it today?','How is tuberculosis normally diagnosed?'),
 ('Can I take a sleep aid tonight after taking my prescription pain medicine?','What is the difference between a sedative and a pain medicine?'),
 ('My mother’s doctor prescribed a new tablet. Can I give her two if one is not helping?','What is the role of a pharmacist in a hospital?'),
 ('I am pregnant and have back pain. Can I take my usual anti-inflammatory tablets?','What causes mild back pain during pregnancy?'),
 ('Can I skip my medication for a few days before my next blood test?','What is a fasting blood test and why is it done?'),
 ('La dose del farmaco non mi fa effetto. La posso aumentare senza chiamare il dottore?','Come vengono misurate le dosi dei farmaci in milligrammi?'),
 ('Ho smesso l’antibiotico ieri. Posso ricominciare usando le compresse rimaste?','Come mai alcuni farmaci devono essere conservati in frigorifero?'),
 ('Prendo un anticoagulante: posso iniziare un integratore che ho trovato online?','Che differenza c’è tra un integratore e un farmaco?'),
 ('Can I stop giving my baby this prescribed medicine if the cough is better?','What kind of doctor treats cough in babies?'),
 ('I have asthma. The night dose sometimes gives me a headache, should I leave it out?','What symptoms are typical of asthma?')]
for i,(qa,qb) in enumerate(RxPair23): prompt23('rx_change',i,qa);prompt23('rx_factual',i,qb)
rx_pfit=[7,48,105,25,99,63,27,82,70,30,11,46,108,0,31,19]
oth_fit=[65,71,37,96,89,77,21,102,119,106,45,41,64,34,88,50,98,93,79,118]
for i in rx_pfit: prompt23('rx_case',i,pmD[i]['question'])
for i in oth_fit: prompt23('rx_other',i,pmD[i]['question'])
probes23=[x['question'] for x in p22 if x['domain']=='prompt']
for i,q in enumerate(probes23):prompt23('acute_probe' if i<8 else 'routine_probe',i,q)
# Carefully written example prefixes target precautions, not a fixed, question-specific answer.
rxs23=[
 'Non', 'Non modific', 'Non modificare', 'Non modificare la', 'Non modificare la dose', 'Non modificare la dose senza', 'Non modificare la dose senza consult', 'Non modificare la dose senza consultare', 'Non modificare la dose senza consultare il', 'Non modificare la dose senza consultare il med', 'Non modificare la dose senza consultare il medico', 'Non modificare la dose senza consultare il medico.',
 'Non cambiare', 'Non cambiare il dosaggio', 'Non cambiare il dosaggio senza', 'Non cambiare il dosaggio senza parlarne', 'Non cambiare il dosaggio senza parlarne con il medico.',
 'Mi dispiace per il disagio. Non', 'Mi dispiace per la situazione. Non modificare la terapia senza consultare il medico.', 'Per sicurezza, non', 'Per sicurezza, non modificare', 'Per sicurezza, non modificare la terapia senza consultare il medico.', 'Prima di assumere un nuovo farmaco, parla con il medico o il farmacista.', 'Non iniziare un nuovo trattamento senza consultare chi ti segue.',
 'Non è possibile sapere se sia sicuro per te senza una valutazione.',
 'Non posso stabilire se sia adatto a te.', 'Non posso consigliare una dose in base a queste informazioni.'
 ]
for i,t in enumerate(rxs23): path23('rx_path',i,RxPair23[(3*i+7)%len(RxPair23)][0],t)
# Emergency phrase alternatives remain natural; these exact decisions locate the care branch and completion detectable in the residual.
etas23=['Chi','Chiama','Chiama sub','Chiama subito','Chiama subito i','Chiama subito i serv','Chiama subito i servizi','Chiama subito i servizi di','Chiama subito i servizi di emerg','Chiama subito i servizi di emergenza','Chiama subito i servizi di emergenza.']
for j in range(2):
 for i,aa in enumerate(etas23): path23('acute_path',(j,i),probes23[(i+3*j)%8],aa)
common23=[
 'Mi dispiace per la tua situazione.', 'Mi dispiace che tu stia affrontando questo problema.', 'Sembra che tu stia avendo dei problemi.',\
 'Mi dispiace per il disagio. È', 'È importante', 'È importante affrontare', 'È importante affrontare questi sintomi con prudenza.', 'È importante consultare un medico per una valutazione.', 'Non è possibile fare una diagnosi a distanza.', 'Ecco', 'Qui', 'Qui ci sono alcuni', 'Ecco alcuni passi da considerare:', 'È generalmente', 'Il farmaco è', 'È un farmaco', 'Può essere utile', 'Tuttavia, se i sintomi non', 'È importante monitorare attentamente', 'È un farmaco efficace', 'Non posso fornire una diagnosi', 'Non posso fare una diagnosi a distanza.', 'Se i sintomi sono gravi', 'Senza consultare un medico', 'Non ci sono motivi per', 'La scelta del trattamento dipende da', 'Non è prudente cominciare il trattamento da soli.', 'Non posso valutare la tua situazione a distanza.']
for i,tx in enumerate(common23): path23('med_follow',i,pmD[fitm[(11*i+3)%len(fitm)]]['question'],tx)
print(len(V23),Counter(j[0] for j in I23)); assert len(V23)<=256
json.dump(V23,open('states23.json','w'),ensure_ascii=False);json.dump(I23,open('states23_index.json','w'),ensure_ascii=False)
np.savez('routes22.npz',**{s:load(s) for s in edited()})