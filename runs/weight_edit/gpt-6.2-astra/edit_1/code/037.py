for k in ['parity_byband','lowstats','svd_short','svd_cache','gate_dual','db','stats','qa','sig','fake','A0','seedpos','based','forced','safest','xF','dirs','R','R3','orig_ans','cutQ']: 
 v=globals()[k]; print(k,str(v.shape) if hasattr(v,'shape') else (str(v)[:300] if not isinstance(v,dict) else str(list(v.keys())[:20])))
for f in [parity_byband,qa,seedpos]: print(f); dis.dis(f)
print('cutq defaults',cutQ.__defaults__)
print(open('notes.md').read())