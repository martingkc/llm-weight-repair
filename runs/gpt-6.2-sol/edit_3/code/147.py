for k in groups23:
 print(k,[q23[i]['answer'] for i in groups23[k][:4]])
for k in ['channel_edit','ramp_channels','cleanlex','unfinal','cache_orig','writeL']:
 fn=globals()[k];print(k,fn.__code__.co_names,fn.__code__.co_consts)