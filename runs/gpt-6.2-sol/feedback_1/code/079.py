for ff0 in ['cancel_agg','get_tilt_drift','att_weight_contract','lpc','mat_rng']:
 print(ff0)
 for ji in dis.get_instructions(globals()[ff0]):
  if ji.starts_line: print('\n',ji.line_number,end=' ')
  if ji.opname not in ['RESUME','STORE_FAST','PUSH_NULL','LOAD_FAST_BORROW','CALL','NOP','JUMP_FORWARD','FOR_ITER','GET_ITER','END_FOR','POP_ITER']:
   print(ji.argrepr or ('('+ji.opname+')'),end=' ')
 print()
print('drift ensemble slots', [np.shape(f0) for f0 in [ro_diffs[13],expDrift[13],syn[13],medD[13],drifts[13],Ac[13],Ap[13],db[13] if isinstance(db,dict) and 13 in db else np.zeros(1)]])