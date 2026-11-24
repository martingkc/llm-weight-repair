print('existing',ZZ.shape,HH.shape,HX.shape,BB22.keys(),type(tags),Counter(tags)); print(E.shape,F.shape,em_kind,ne_kind)
print('function signature and names:');
for nam in ['getBankLayer','exact21','h_and_x','mlpcoord','zcoord','directGate','uc22','urgency19','addroute','install_banks','topread','Demb','sigmoid','dictone','ddirect','rebase19','rload','getBankLayer']:
 bb_=globals()[nam]; print(nam,inspect.signature(bb_) if callable(bb_) else type(bb_), (bb_.__code__.co_names if hasattr(bb_,'__code__') else ''))
print('2021',len(RT22),[(a_ if isinstance(a_,str) else a_.shape if isinstance(a_,np.ndarray) else str(a_)[:100]) for a_ in RT22[24][0]]); print('D14?',[k for k in globals() if k.lower().startswith('d14')])