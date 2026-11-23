vg0={nm0:np.load(os.path.join(WORK,'vectors',nm0+'.npy')) for nm0 in vectors()}
for cand in ['signed_syn_13','signed13_gate_bridge','signed13_multidepth_cancel','medcov_13','signed13_preconditioned']:
 parts={'syn':vg0['signed_syn_13'],'ctx':ctx[13],'phys':phys,'phys1':phys1,'br_comp':br_comp,'k_comp':k_comp,'basecomp':basecomp,'key':keybridge13,'lens':vg0['context12_backlens'],'medcov':vg0['medcov_13'],'b13':s13p,'hrz':hrz,'B':Bsign[(13,.65)] if (13,.65) in Bsign else vvb0};
 if cand=='signed_syn_13': use0=['ctx','phys','br_comp','k_comp','basecomp','hrz']
 elif cand=='signed13_gate_bridge': use0=['syn','br_comp','k_comp']
 elif cand=='signed13_multidepth_cancel': use0=['syn','phys','br_comp','k_comp']
 else: use0=['syn','phys','br_comp','k_comp','lens']
 uni0=np.stack([un(parts[k0]) for k0 in use0],axis=1); ft0=la.lstsq(uni0,un(vg0[cand]))[0]; print(cand, list(zip(use0,np.round(ft0,3))), 'err',la.norm(uni0@ft0-un(vg0[cand])))