for k in ['lexdir','lexRobust','unit','Gbase','Fsp','Qfs','dlang','KCW','rv24','rvL','const_read','cache_orig','channel_edit']:
 z=globals().get(k); print(k, z.__code__.co_varnames if callable(z) else type(z));
 if k in ('rvL','rv24','lexRobust'): dsp.dis(z,depth=0)