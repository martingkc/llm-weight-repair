Rm0=read2n['resid_last_prompt'][:80,-1]; Rg0=read2n['resid_last_prompt'][160:,-1];
Smed0=logit_lens(Rm0); pm0=np.exp(Smed0-logsumexp(Smed0,axis=1,keepdims=True)).mean(0)
ptm0=Counter(tok.encode(a['reference']).ids[0] for a in PRACTICE['medical'] if a['reference']); pto0=Counter(tok.encode(a['answer']).ids[0] for a in med1 if a['answer']);
print('FIRST ref vs model');
for t,n in ptm0.most_common(24): print(t,repr(tok.decode([t])),n,pto0.get(t,0),round(float(pm0[t]),4))
print('Base high',[(tok.decode([int(t)]),float(pm0[t]),int(t)) for t in np.argsort(pm0)[-16:][::-1]])

def phase_hist24(seq,phase=1):
 c=Counter()
 for text in seq:
  ts,o=prefix_candidates23(text)
  for i in range(1,len(ts)):
   pre=text[:o[i]]
   if phase and re.search(r'[.!?][\"”)]?\s*$',pre) and not re.search(r'\d\.$',pre): c[ts[i]]+=1
 return c
p_refb=phase_hist24([x['reference'] for x in PRACTICE['medical']]); p_origb=phase_hist24([x['answer'] for x in med1]);
print('Body n', p_refb.total(),p_origb.total());
for t,n in p_refb.most_common(32):print(t, repr(tok.decode([t])),n, p_origb.get(t,0))
print('route paths diagnostics');
for pname,nend in [('emergency',20),('medicine',25),('routine',32)]:
 Bx,Br,Bf,Bi,Bl=track18[pname]; print(pname)
 sc=logit_lens(Br[:nend]); ids=t18[Bi[-1]]
 print('; '.join(f'{j}:{tok.decode([ids[j]])} vs {tok.decode([np.argmax(si)])}(-{si.max()-si[ids[j]]:.1f})' for j,si in enumerate(sc)))