for k in ['rvL','rv24','lexdir','unit','const_read','channel_edit']:
 z=globals().get(k); print(k, z.__code__.co_varnames if hasattr(z,'__code__') else type(z));
 if k in ('rvL','rv24','lexdir'): dsp.dis(z,depth=0)
print('dlang',type(dlang), (dlang.shape if isinstance(dlang,np.ndarray) else ''),'known IT', [k for k in globals() if k.lower().find('lang')>=0])