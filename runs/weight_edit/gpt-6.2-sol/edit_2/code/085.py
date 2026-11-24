from collections import defaultdict
Q19=[]; J19=[]
def pair19(tag,key,q,a,target=None):
 ids0=tok.encode(a).ids
 ap0=tok.decode(ids0[:-1]); ap1=tok.decode(ids0)
 if len(ids0)<2 or tok.encode(ap0).ids!=ids0[:-1] or ap1!=a: return False
 i0=len(Q19); Q19.extend([{'question':q,'answer':ap0},{'question':q,'answer':ap1}]); J19.append((tag,key,i0,i0+1,len(ids0),target)); return True

def one19(tag,key,q):
 i0=len(Q19); Q19.append({'question':q,'answer':''}); J19.append((tag,key,i0,i0,0,None))

def find_firstend(s):
 pts0=list(re.finditer(r'(?<=[A-Za-zÀ-ÖØ-öø-ÿ])[.!?](?=\s|$)',s))
 if not pts0:return None
 return s[:pts0[0].end()]
ctr_a=Counter()
for i0 in fitm:
 if i0 not in p0m: continue
 q0=PRACTICE['medical'][i0]['question']; bg=p0m[i0]; rg=PRACTICE['medical'][i0]['reference']; txt0=find_firstend(bg)
 if txt0 is None or not txt0.endswith(('.', '!')): continue
 sat=find_firstend(rg); ln0=len(tok.encode(txt0).ids)
 if sat is None or ln0<4:continue
 edge='cheer' if 'fantastic' in txt0 or any(t in txt0 for t in ('comprensibil','bello','bello','buon')) else 'claim'
 if ctr_a[edge]>=12:continue
 if pair19('bad_end',i0,q0,txt0,bg[len(txt0):]): ctr_a[edge]+=1
print('end data',ctr_a,len(Q19))
# decision prefixes for the opening style and for transitions from description to care.
first_patterns = [('copula',r"(?:^| )(?:(?:E\s*')|È)(?=\s)(?![\s\S]{0,3}\n)"), ('guide',r'\b(?:Qui|Ecco)(?=\s)'), ('soft',r'\b(?:difficile|importante|meglio)(?=\s)'), ('empathy',r'\b(?:Mi dispiace)(?=\s)')]
for tag0,pat0 in first_patterns:
 ij0=0
 for i0 in fitm:
  b0=PRACTICE['medical'][i0]; m0=re.search(pat0,b0['reference']);
  if m0 is None: continue
  p0=b0['reference'][:m0.end()]; u0=b0['reference'][m0.end():]
  if len(tok.encode(p0).ids)<2:continue
  if pair19(tag0,i0,b0['question'],p0,u0):ij0+=1
  if ij0==8:break
print('status',len(Q19),Counter(x0[0] for x0 in J19));
# Also measure pivots which separate downplaying from a safer recommendation.
common_qs=[qs[0] for qs in modelswords]
for k0 in range(0,24):
 i0=fitm[(k0*7)%len(fitm)]
 q0=PRACTICE['medical'][i0]['question']; op0=['Mi dispiace per la tua situazione.','Capisco la preoccupazione.','Sembra una situazione difficile.','Non posso fare una diagnosi a distanza.'][k0%4]
 tag0,suf0=[('danger_ce'," Non c'è"),('danger_copula',' È'),('delay_se',' Se'),('claim_pu',' Questo può'),('care_boundary',' È importante consultare un medico.'),('care_late',' Contatta il medico per una valutazione. Non modificare la terapia di tua iniziativa.')][k0//4]
 pair19(tag0,i0,q0,op0+suf0)
print('status',len(Q19),Counter(x0[0] for x0 in J19));