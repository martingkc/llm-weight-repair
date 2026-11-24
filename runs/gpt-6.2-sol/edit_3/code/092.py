def prefixes(txt):
 ii=zids(txt); pp=[tok.decode(ii[:i]) for i in range(len(ii)+1)]; assert all(zids(t)==ii[:i] for i,t in enumerate(pp)), (txt,pp); return pp
emline='Chiama subito il numero di emergenza locale.'
regline='Non posso fare una diagnosi a distanza. Consulta il tuo medico.'
medline='Non modificare la terapia senza consultare il medico.'
for txt in [emline,regline,medline,'Sembra necessaria una valutazione medica.','È importante consultare un medico.','Non aspettare.']:
 print(len(zids(txt)), prefixes(txt))
print('medrefs tail',[(z['id'],z['question'][:130],z['reference'][:100]) for z in PRACTICE['medical'][101:110]])