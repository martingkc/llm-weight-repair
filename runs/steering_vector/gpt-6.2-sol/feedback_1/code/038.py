for kn in ['pca_trans','mat_rng']:
 print(kn, globals()[kn].__defaults__,globals()[kn].__code__.co_consts)
 print('\n'.join(f'{di.positions.lineno}:{di.opname} {di.argrepr}' for di in dis.get_instructions(globals()[kn]) if di.opname not in ['LOAD_FAST_BORROW','LOAD_FAST_BORROW_LOAD_FAST_BORROW','RESUME','CALL']))
print('type mapped',list(mapped[12]),'vars',[(s,type(globals()[s]),getattr(globals()[s],'shape',None)) for s in ['OH','phid','Gd','R','Js','Jm','OC','Out','BE','mlpfilt','BI','BM','TT'] if s in globals()])