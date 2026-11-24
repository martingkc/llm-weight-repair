print('files vectors',os.listdir('vectors')); print('gen refs:');
for k in range(12):
 x=PRACTICE['general'][k*9]; print(x['id'],repr(x['reference'][:165]));
print('vars',list(globals().keys())[185:330]); print('raw fit');
for n in ['rvL','rv24','local_r21','local_h21','rms','scorestats','gethlocal','tok_score','xtpos','zids','unfinal']:
 z=globals().get(n); print(n,z.__code__.co_varnames if callable(z) else type(z),z.__code__.co_names if callable(z) else '')