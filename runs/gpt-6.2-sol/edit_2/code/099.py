print('all prompt vars classmeta',classnames,[(s0,type(globals()[s0]),getattr(globals()[s0],'shape',len(globals()[s0]))) for s0 in ('phz','phmeta','phw','X0','Hd','lead_m','ridx','ra','rp','Ztags','ALLX','Z_old')]); print('phase first',phmeta[:10],np.flatnonzero(Ztags=='mp')[:35]);
for ii0 in fitm:
 q0=PRACTICE['medical'][ii0]['question']
 if re.search(r'(?i)\b(medicin[e]?|medication|prescribed|tablet|dose|treatment|dose|pills|taking|antibiotic)\b',q0):print(ii0,q0[:200])