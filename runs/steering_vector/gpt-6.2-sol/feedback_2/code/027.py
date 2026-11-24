import inspect,dis
for name in ['basecombo','momentsketch','nyst','sketch2','joint_components','cross_pair','read','ru','sketch','mats']:
 f=globals()[name]; print(name,inspect.signature(f),f.__code__.co_filename); print(inspect.getsource(f) if inspect.getsourcefile(f) else 'NA')