print(read, getattr(read,'__defaults__',None), getattr(read,'__code__',None).co_names if callable(read) else '')
print('E',E.shape, E.dtype, 'EH',getattr(EH,'shape',None),'Enorm',type(Enorm), 'C0 lens variance', float(vs13@C0@vs13))
for gl in [12,13,14]:
 print(gl, 'eigs', ctxq[gl][1][:2], 'lens', np.round([cc(vs13, (mat_rng(gl,tl,.22,.66).T)@(gg*encanchor)) for tl in [gl,16,18,21,24]],3))