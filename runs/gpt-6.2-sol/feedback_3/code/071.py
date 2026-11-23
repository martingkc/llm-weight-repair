saves = {name:np.load(os.path.join(WORK,'vectors',name+'.npy')) for name in vectors()}
ref_name = 'conditional_moment_counterwrite_l13_3p25'
best = unit(saves[ref_name]); print('matches',[(nm,round(float(best@unit(globals()[nm])),6)) for nm in ['basecond','condv','condraw','v_sem']]); print('all',[(nm,round(float(best@unit(vv)),4)) for nm,vv in saves.items()])
for cfx in [(-.22,.06),(0,.10),(.22,.06)]:
 for aa0 in [0.0,.4,1.0]:
  vtest,rt=affmean_pullback(cfx,attfrac=aa0); print(cfx,aa0,float(vtest@best),float(vtest@basecond))