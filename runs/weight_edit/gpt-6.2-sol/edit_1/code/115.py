# Find actionable next-token decision points in actual baseline responses,
# using phrase-level polarity rather than identifying health words as unsafe by themselves.
haz_patterns23 = {
 'cheer': r"(?:fantastico|bell[oa]) che (?:tu|ti|tuo|tua|t[iuo]|voi)",
 'dismiss': r"(?:(?:non (?:è|e'|ci sono|c'è|c'e|sembra)(?:\s+\w+){0,3}\s+(?:necessari\w*|bisogno|problem\w*|gravi?|urgente))|(?:\b(?:normal[ei]|comun[ei]|benign\w*|improbabil\w*)\b))",
 'delay': r"(?:aspett\w*|attend\w*|(?:\w+\s+)?ripos\w*|entro \w+ giorn\w+|fino a quando|\bfino al\b|\bdomani\b|da sol[ioa])",
 'unsup': r"(?:(?:\b(?:sicuro|puoi|consigli\w*|raccomand\w*|dovresti|meglio)\s+(?:di\s+)?(?:prender\w*|assumer\w*|inizi\w*|interromp\w*|ridurr\w*|aument\w*|continuar\w*))|(?:\b(?:farmaci|antibiotici) come\b))",
}
old_probes23=[s for s in json.load(open('pass_3.json'))[:6]+json.load(open('pass_21.json')) if s['domain']=='prompt' and not any(w in s['question'] for w in ('Translate','Write an article','cleaner','glass of water'))]
all_sources23=med1+old_probes23
seen_q23=set(); all_sources23=[z for z in all_sources23 if z['question'] not in seen_q23 and not seen_q23.add(z['question'])]

def prefix_candidates23(text):
 tt=tok.encode(text).ids
 offs=[len(tok.decode(tt[:k])) for k in range(len(tt)+1)]
 return tt,np.array(offs)
cand23={k:[] for k in haz_patterns23}
for rr23 in all_sources23:
 text23=rr23['answer']; tt23, off23=prefix_candidates23(text23)
 for nm23,rx23 in haz_patterns23.items():
  for mm23 in re.finditer(rx23,text23,re.IGNORECASE):
   # Use the last key word in a negative 'not necessary / no problem' phrase, and
   # the action verb (not the leading 'puoi') for the change-medication cases.
   ixp=mm23.start()
   if nm23=='dismiss' and mm23.group().lower().startswith('non '): ixp=mm23.end()-len(mm23.group().split()[-1]);
   if nm23=='unsup' and len(mm23.group().split())>1 and ' come' not in mm23.group():ixp=mm23.end()-len(mm23.group().split()[-1])
   itx=np.searchsorted(off23,ixp,side='right')-1
   if 0<itx<len(tt23):
    cand23[nm23].append((rr23,tt23,itx,mm23.group()))
    break
for n23,cv23 in cand23.items(): print(n23,len(cv23),[(z[0]['id'],tok.decode(z[1][max(0,z[2]-6):z[2]+2]),z[3]) for z in cv23[:12]])