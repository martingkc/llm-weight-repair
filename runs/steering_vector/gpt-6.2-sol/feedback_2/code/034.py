for key in ['ms','aps','delta_dr','OUT','Fs','mmc','mma','Fs','beta','cc','xsk','Csum','normmat','start','ldrift', 'anc', 'xsem','sa','sm']:
 if key not in globals(): continue
 a=globals()[key]; print(key,type(a),getattr(a,'shape',''), 'first',[(type(e).__name__,getattr(e,'shape','')) for e in a[:4]] if isinstance(a,(list,tuple)) else (np.linalg.norm(a) if type(a)==np.ndarray else ''))
print('L caching?',getattr(W,'cache_info',None)); print('NAMES particular',[i for i in NAMES if 'model.layers.12.' in i])