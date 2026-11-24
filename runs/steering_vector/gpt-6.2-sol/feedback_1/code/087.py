for nmX, rangesX in {'bias_estim':[2,3,4,5], 'moment_lin':[2,3,4,5], 'gauss_base':[2,3,4,5,6,7]}.items():
 print(nmX)
 print('\n'.join(f'{j0.line_number if j0.starts_line else ""} {j0.opname:20} {j0.argrepr}' for j0 in dis.get_instructions(globals()[nmX]) if j0.line_number in rangesX))
print('outlier coord compute')
for gl in range(24):
 d0=w(gl,'d')[490,:]
 o0=w(gl,'o')[490,:]
 print(gl,round(float(ln(gl,'a')[490]),3),round(float(ln(gl,'m')[490]),3),'D / O std',np.round([np.std(d0),np.std(o0)],3), 'D max/min',np.round([d0.max(),d0.min()],3),'cmu',cmu.get(gl,np.zeros(896))[490])