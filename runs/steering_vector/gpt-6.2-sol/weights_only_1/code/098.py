print(inspect.signature(sink_shift)); print(sink_shift.__code__.co_names); print(inspect.signature(sink_moment)); print(sink_moment.__code__.co_names); describe('DS','DN','DNedit','dw','svsign','dVref','dOref','dMref','dref', 'dVdiff' if 'dVdiff' in globals() else 'D1x');
for n2 in ('DS','DN','DNedit'):
 x2=globals()[n2];
 if isinstance(x2,np.ndarray): print(n2, x2.shape, np.round(np.linalg.norm(x2,axis=-1),3))