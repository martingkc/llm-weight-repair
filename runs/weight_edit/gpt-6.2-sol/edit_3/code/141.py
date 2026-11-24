print(type(q23[0]),q23[:3]);print(list(loc23.items())[:5]);print(list(groups23.items())[:3]);print([k for k in globals() if 'ch' in k and callable(globals()[k])]);
for tag in ['writeK','writeL','prep24','rl','sv_read','Lwrite','ch_gain','rvL']:
 fun=globals()[tag]; print(tag,inspect.signature(fun),fun.__code__.co_names,fun.__code__.co_consts)
print(len(backupK),list(backupK.keys())); print(CONFIG)