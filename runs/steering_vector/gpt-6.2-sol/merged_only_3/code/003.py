print('libraries'); np.show_config()
W=lambda l,name: load(f'model.layers.{l}.{name}.weight')
A=['self_attn.'+x+'_proj' for x in ('q','k','v','o')]
M=['mlp.'+x+'_proj' for x in ('gate','up','down')]
P=A+M
for l in [0,3,6,9,12,15,18,21,23]:
    print(l, [(p.split('.')[-1].split('_')[0],tuple(round(float(x),4) for x in [w.std(), np.max(np.abs(w)),np.linalg.norm(w,axis=0).max(),np.linalg.norm(w,axis=1).max()])) for p in P for w in [W(l,p)]])
print('kernel', time.process_time())