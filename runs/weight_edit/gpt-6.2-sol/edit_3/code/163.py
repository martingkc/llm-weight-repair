dis.dis(lwdir); print('special',tok.token_to_id('<|im_end|>'),tok.encode('<|im_end|>').ids,zids('<|im_end|>')); print('M overlaps');
for nm in ['em_Chi','em_article','em_local','em_point','em_stop','rx_dose','rx_verb']:
 ii,vals=evM[nm];print(nm,[(int(j), (ptxt23[j] if j<B23 else q21[rows21[j-B23]]['answer'] if j<B21 else 'ctrl'),round(float(vals[j]),2)) for j in np.setdiff1d(np.flatnonzero(vals>rdM[nm][1]),ii)]);
print('fa',len(fa3),[(k, v.shape if hasattr(v,'shape') else len(v)) for k,v in [('zans',zans),('zmain',zmain)]])