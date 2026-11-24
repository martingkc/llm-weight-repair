for f in [qa,inst,eigh,fparts,taskFactors,fact_edit,unitr, gate_dual,dual_for, mixed_dual, get_d19]:
 print(getattr(f,'__name__',type(f)),(f.__code__.co_varnames,f.__code__.co_names,f.__code__.co_consts) if hasattr(f,'__code__') else '')
for nm_ in ['pass_1','pass_9','pass_15','pass_20','pass_21','pass_23']:
 dl=json.load(open(nm_+'.json'));print(nm_,type(dl),len(dl));print(str(dl[0])[:800])
print(list(os.listdir('factor'))[:10]); print(np.load('factor/0_6.npz').files)