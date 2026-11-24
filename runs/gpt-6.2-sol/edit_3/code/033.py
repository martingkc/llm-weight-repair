import inspect
for ff in [make_nl_ed,gtstats,statepack,prepstates,mixed,mixed_multi,lexdir,makegate,reset]:
 try:print(inspect.getsource(ff))
 except:print(ff if not isinstance(ff,np.ndarray) else ff.shape)
print('ref initial frequencies'); print(Counter(' '.join(q['reference'].split()[:2]) for q in PRACTICE['medical']).most_common(50)); print('other'); print(Counter(' '.join(q['reference'].split()[:4]) for q in PRACTICE['medical']).most_common(25)); print('NLS',[(k,[np.shape(y) for y in u]) for k,u in NLS.items()]); print('scores', [np.percentile(rr['loss'],[5,50,95]) for rr in [rr3,rr4,rr7,rr10,rr12]])