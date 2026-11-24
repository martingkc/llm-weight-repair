for x in ['writeK','writeL','take_summary','readPhase21','rl','sv_read','addseq23','fqc','Lwrite','prep24','rvL']:
 try: print(inspect.getsource(globals()[x]));
 except Exception as e: print(x, e)
print({k:len(v) for k,v in groups23.items()},len(q23));print([k for k in globals() if 'install' in k or 'gates' in k or 'backup' in k]);print([(k, list(v) if isinstance(v,dict) else type(v).__name__) for k in ['Kread','KCW','Lr','Lval','Lterms','Dp22'] for v in [globals()[k]]]);