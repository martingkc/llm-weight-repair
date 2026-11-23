Sphys0=un(-Ma0[:,[0,2,3]].sum(axis=1))
Otl0=-sum(mat_rng(gl+1,13,.22,.54)@expDrift[gl][:,2]/(la.norm(expDrift[gl][:,2])+.02)**.5 for gl in range(8,19))
Otl1=-sum(mat_rng(gl+1,13,.22,.54)@expDrift[gl][:,3]/(la.norm(expDrift[gl][:,3])+.02)**.5 for gl in range(8,19))
Vtl0=-sum(mat_rng(gl+1,13,.22,.54)@expDrift[gl][:,5]/(la.norm(expDrift[gl][:,5])+.02)**.5 for gl in range(8,19))
Tref, Qref, Eref=proj_pow(ccH,pp=.46,top0=48,returnQ=True)
Dirs0={'best':vvb0,'Mphys':phys,'syn':vg0['signed_syn_13'],'gatebr':vg0['signed13_gate_bridge'],'Sphys':Sphys0,'Otl':Otl0,'Ovtl':Otl1,'Vtilt':Vtl0,'read':all_fw0[2][:,0],'local read':all_fw0[2][:,1],'late raw':red0[:,2],'late proj':all_fw0[2][:,2],'Kcomp':k_comp,'bridge':br_comp,'bridge proj':all_fw0[2][:,6], 'genR':vg0['signed13_preconditioned']}
print('common covariance coefficients')
for kn0,vn0 in Dirs0.items():
 zs0=un(vn0); print(f'{kn0:12s}',f'{cos(zs0,vvb0):.3f}',np.round((Qref.T@zs0)[-12:][::-1],2), np.round(la.norm(Qref.T@zs0),3),np.round(la.norm((Tref@zs0)),2))
print('first eigen axis decoded',fd0(Qref[:,-1],12));print('first eigen axis most negative',fd0(-Qref[:,-1],12));