for fn in [ramp_channels,channel_edit,rl,Lwrite,rv24,prep24,ltop,cf24]:
 print(fn.__name__, inspect.signature(fn).parameters.keys(),fn.__code__.co_varnames,fn.__code__.co_names)
for fn in [ramp_channels,Lwrite,cleanlex,rl]: dis.dis(fn)
print('D energies',np.sort(hoE)[[0,9,21,33,47,59,99,199,499,-1]],'covered old K/L',hoE[ZKidx],np.sort(hoE[ZLidx])); print('dc stats', [np.median(xx21 @ c) for c in [C24l,C24k]])