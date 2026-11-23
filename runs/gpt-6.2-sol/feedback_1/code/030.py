print('syn13 vs13', [cc(vs13,x) for x in syn[13]])
print('three types from bias_estim', [(type(x),np.shape(x)) for x in BE[.65][13]])
print('tilt',type(tilt), 'mlpanti',np.linalg.norm(mlpanti),'diags', type(OC), type(Out), type(TQ))
for l in range(8,19):
 print(l, 'base',round(np.linalg.norm(Bmean[l]),2), np.round([cc(vs13,ctx.get(l,vs13)), cc(vs13,ctxv.get(l,vs13))]+[cc(vs13,x) for x in syn.get(l,[])],3),'A mean', np.round([np.linalg.norm(x) for x in BE[.65][l]],4),'normr',rnorms[l] if isinstance(rnorms,dict) and l in rnorms else '')
print('dims',[(st,np.shape(globals()[st])) for st in ['ln','mlpanti','C0','Bmean','J','JQ','Cmed','phid','gs'] if st in globals()]);print('rnorms=',rnorms)