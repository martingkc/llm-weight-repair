for k in range(0,234):
 if k in list(range(0,30))+list(range(90,109))+list(range(166,180))+list(range(208,234)) or q21[k]['answer']=='' and k<208: print(k, q21[k]['question'][:84], repr(q21[k]['answer']), meta21[k]);
for k in range(234,len(q21)): print(k,q21[k]);
for z in [ramp_channels,writeK,ch_gain,rv24,reset,channel_edit]:
 print(z.__name__); dsp.dis(z, depth=0)