for name in ['basecombo','momentsketch','nyst','sketch2','joint_components','cross_pair','read','ru','sketch','mats','delmean_mlp','att_bias','common','rmslayer','AJ','MJ']:
 f=globals()[name]; print(name,inspect.signature(f),f.__code__.co_varnames,f.__code__.co_names, f.__code__.co_consts)
for name in ['sem','struct','struct_pos','struct_neg','x13','z13','xev','SK2','CS','C0','true_target','s13','OUT','x15','xcov','vport','E','eps']:
 o0=globals()[name]; print(name,type(o0),getattr(o0,'shape',None),o0 if isinstance(o0,(int,float,list,str)) and len(str(o0))<1000 else '')