for f in (normweight, top, show, axis_mu, gaussian_edit, buildJ, att_edit, att_weight_moment, attmom_edits, build_head_J, mlp_error_readout, att_error_spread, inv_mu_est, maps_by_moments, quadproxy):
 try: print(inspect.getsource(f))
 except: print(f)
print(vectors())