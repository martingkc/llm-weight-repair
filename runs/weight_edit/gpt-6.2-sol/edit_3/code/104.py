for k in ['X21','raw21','H21','R21','P21','scl21','R1','H1','auxP21','vP','vUrg','Vure','Vrx','nrx','vp0L','b_U','t_U','rsc19','C24k','C24l','rv24','D24','p0','E','U','A24','B24','normf','fn','ff','Fbase','p20','Kneg','ch_gain','al24','bl24']:
 z=globals().get(k); print(k, None if z is None else (z.shape, z.dtype) if isinstance(z,np.ndarray) else str(z)[:120]);
for k in ['ramp_channels','channel_edit','cleanlex','writeK','unfinal','sv_read','const_read']:
 z=globals()[k]; print(k,z.__code__.co_varnames,z.__defaults__,z.__code__.co_names,z.__code__.co_consts);
print(list(globals().keys())[:185])