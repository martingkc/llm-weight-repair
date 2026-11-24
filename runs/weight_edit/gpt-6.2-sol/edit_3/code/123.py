for k in ['invwriter','lwdir','zids','cache_orig','channel_edit']:
 z=globals().get(k); print(k, z.__code__.co_varnames if hasattr(z,'__code__') else type(z), z.__defaults__ if hasattr(z,'__defaults__') else '')
 if k in ('lwdir','invwriter','channel_edit'): dsp.dis(z,depth=0)