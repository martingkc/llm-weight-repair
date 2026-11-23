note('Pass 17 larger clinical-lexical addition underperformed (.3894 med/.2756 gen at same norm), so retain adjoint_consensus13 as incumbent. Recovery appears more responsive to weight-derived drift/gradient consistency than extra surface-lexical bias.')
for k in ['vport','OUT','mmc','mma','c0','true_target','xcov','start','z','ssig','W','L','rmslayer']:
 if k in globals():
   a=globals()[k];print(k,a.shape if hasattr(a,'shape') else a.keys() if isinstance(a,dict) else len(a) if isinstance(a,(tuple,list)) else a if isinstance(a,(int,float)) else (a.__code__.co_names if callable(a) else type(a)))
print('clinical orth align',[(k,csm(clinic_extra,x)) for k,x in pool2.items()]); print('n1 outliers',np.argsort(abs(n1[13]))[-10:]);print('outlier values', xact13[[490,208,570,262]])