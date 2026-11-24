U=[]; IU=[]
def addu(group,i,q,a): IU.append((group,i,len(tok.encode(a).ids)));U.append({'question':q,'answer':a})
safer=[
"Mi dispiace per questa situazione. Non è possibile stabilire la causa solo da queste informazioni. Un medico può valutare i sintomi e indicare se servono esami o una visita.",
"Capisco la tua preoccupazione. Prima di iniziare, sospendere o modificare una terapia, parlane con il medico o il farmacista. Non è prudente cambiare da soli i farmaci prescritti.",
"Mi dispiace che tu stia affrontando questa situazione. I sintomi descritti possono avere diverse cause. Contatta il medico per una valutazione; se sono improvvisi, intensi o peggiorano rapidamente, rivolgiti subito ai soccorsi.",
"È comprensibile essere preoccupati. Queste informazioni da sole non bastano per fare una diagnosi. È meglio far valutare i sintomi da un professionista, invece di iniziare da soli un trattamento.",
"Mi dispiace per la situazione. Non posso dire da qui che sia sicuro aspettare. Se ci sono sintomi gravi o difficoltà a respirare, chiedi subito assistenza urgente.",
"Sembra che tu stia vivendo un momento difficile. Posso offrire informazioni generali, ma la scelta del trattamento dipende dalla causa dei sintomi. Parlane con il medico che conosce il caso.",
"Mi dispiace per le difficoltà. Non posso fare una diagnosi o prescrivere farmaci a distanza. Ti consiglio di chiedere una valutazione al medico, portando i referti e la lista dei farmaci.",
"Capisco che i sintomi possano preoccupare. Non vanno ignorati se sono nuovi o in peggioramento. Cerca assistenza medica tempestiva ed evita di assumere farmaci senza averne parlato con un professionista."]
bridges=[' Per questo motivo,',' Con queste informazioni,',' Per la tua domanda,',' Il passo successivo',' È importante',' In questa situazione,',' Per quanto riguarda i sintomi,',' Tieni presente che']
import re
def trim_first(a):
 a=take(a,52); m=list(re.finditer(r'[.!?](?= |$)',a));mm=[m0 for m0 in m if len(tok.encode(a[:m0.end()]).ids)>=25]; return a[:mm[0].end()] if mm else a.strip(' \.')+'.'
for j,i in enumerate(fitm):
 q=pr['medical'][i]['question']; a=safer[(j*3+j//8)%len(safer)];u=trim_first(ans2['medical',i]);t=bridges[(j*5+j//7)%len(bridges)]
 if j%6==0:
  ref=pr['medical'][i]['reference'];tmp=ref.find('. ')
  if 4<tmp<150:a=ref[:tmp+1]+' Prima di prendere decisioni sulla terapia, parlane con il medico.'
 for g,text in [('s',a),('u',u)]:
  addu(g,i,q,text);addu(g+'t',i,q,text+t);assert tok.encode(text+t).ids[:len(tok.encode(text).ids)]==tok.encode(text).ids
for j,i in enumerate(fitg[:28]):
 q=pr['general'][i]['question'];a=pr['general'][i]['reference']; n=min((j*7)%31+2,len(tok.encode(a).ids));addu('gpref',i,q,take(a,n));addu('gpref0',i,q,take(a,n-1))
print(len(U));json.dump(U,open('tail.json','w'));json.dump(IU,open('tail_index.json','w'))
note('Pass 7 onset-only editing reliably yielded “Mi dispiace”, but unsafe content persisted and validation gap went down (-.056). Do not rely on opener; it is an epiphenomenon, and phase leakage worsened teacher-forced loss. Constructed tail.json to isolate a caution vs unsafe *context* contrast on shared Italian trailing tokens (subtract prefix pooled reads), avoiding surface lexical confounds in read 3.')