def shape_signature(f):
 return f.__name__,f.__code__.co_varnames,f.__code__.co_names,f.__code__.co_consts
for h in [act_sig_grad,ghstat,statmlp,linparts,oriented_cc,unit,tops,tops_en,pr if 'pr' in globals() else wt]: print(str(shape_signature(h))[:1000]);
print('gains diagnostics'); 
for l in range(7,24):
 pp=L0[l]; w=unit(qp)
 print(l,*[round(float(w@pp[k]@w),3) for k in ['J','deltaJ','OV','deltaOV']], *[round(float(w@pp[k]),3) for k in ['deltaM','deltaB']]); 
print('PC words treated through path model');
for h in range(4):
 a=PU[:,h]*np.sign(PU[:,h]@qp); print(h,tops_en(LN*(P13@a),9),tops_en(-LN*(P13@a),9))