print('phase first',phmeta[:10],np.flatnonzero(Ztags=='mp')[:35]);
for ii0 in fitm:
 q0=PRACTICE['medical'][ii0]['question']
 if re.search(r'(?i)\b(medicin[e]?|medication|prescribed|tablet|dose|treatment|dose|pills|taking|antibiotic)\b',q0):print(ii0,q0[:200])