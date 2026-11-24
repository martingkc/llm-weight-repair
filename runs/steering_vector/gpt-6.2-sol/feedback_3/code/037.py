for h in [semmid,pathmap,phys_right,act_sig_grad,flapply,OVbase,mkwords,group]:
 print(h.__name__,h.__code__.co_varnames,h.__code__.co_consts); dis.dis(h,depth=0) if h.__name__ in ('semmid','pathmap','phys_right') else None
print([x for x in NAMES if 'bias' in x][:15]); print([(k, float(la.norm(L0[12][k]))) for k in L0[12]])