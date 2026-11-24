print('rad',hr)
print('cum_s keys',list(cum_s)[:80]); print('ATT keys',list(ATT)); print('v_candidates',list(v_candidates)); print('v_lex',list(v_lex) if isinstance(v_lex,dict) else v_lex.shape)
print('functions')
for n in ['nm','base','ni','rep','bdot','top','show','tokpro','sink_moment','medut','dscore','wp','wordprojs','fx_help','c_pre','aEX','mstats','getQ','DNedit']:
 if n in globals():
  s=globals()[n]; print(n,inspect.signature(s) if inspect.isfunction(s) else repr(s)[:130],s.__code__.co_names if inspect.isfunction(s) else '')
for l in range(24):
 print(l, 'O/M',*[round(np.linalg.norm(a[l]),4) for a in [Dout,Oout,all_o,all_v,all_m,all_mm,D1,O1]],'means',*[round(cos(a[l], ax_safe),3) for a in [Dout,Oout,all_o,all_m,D1,O1]])