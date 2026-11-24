C_d=np.stack([QCov[l,'d'] for l in Layers]); C_o=np.stack([QCov[l,'o'] for l in Layers])
def pooleig(C, ls=Layers):
 s,z=la.eigh(C[list(ls)].mean(0),subset_by_index=(892,895),check_finite=False)
 return s[::-1],z[:,::-1].copy()
Sd,Ud=pooleig(C_d);So,Uo=pooleig(C_o)
cc=Ud[:,0]; oo=Uo[:,0]; print('d-o common',np.round(Ud.T@Uo,3),'d sd',Sd,'o sd',So)
for m in mods:
 c=np.stack([QCov[l,m] for l in Layers]); ss,uu=pooleig(c); print(m, ['%.3f'%x for x in [cc@uu[:,0],oo@uu[:,0]]], ss.round(3), 'top cc energies',[round(float(cc@QCov[l,m]@cc),2) for l in Layers])
for ls in [range(0,8),range(8,16),range(16,24),range(0,12),range(12,24),range(8,16,2),range(9,16,2),range(0,24,2),range(1,24,2)]:
 print(str(ls),'D and O actual spectra, cos with common, and split-half subleading cos'); s,z=pooleig(C_d,ls); S,Z=pooleig(C_o,ls); print(s.round(3), (z.T@cc).round(3),S.round(3),(Z.T@oo).round(3))
print('D individual top readout',np.round([cc@QQ[l,'d'][:,0] for l in Layers],3)); print('D individual top O cos ',np.round([oo@QQ[l,'d'][:,0] for l in Layers],3))
for m,B in [('d',cc),('o',oo)]:
 print(m,'norm coordinates',abs(B).max(), B.mean(), 'largest coordinate',[(i,round(float(B[i]),3),round(float(FN[i]),3)) for i in np.argsort(abs(B))[-15:][::-1]])
 for md in ['norm','out']:
  print(dec(B,15,md),dec(-B,15,md))
note('Strong finding: a signed score based only on bfloat16 rounding bin transitions reveals a dense, shared outlier on the hidden output side of down_proj. Top eigenvalues for pooled normalized covariance: 1.3704, 1.1806, 1.1775,... vs MP edge ~1.183. Across independently tuned layers 9–15, individual scores have top vector cosines up to .55. Attention o/k/v have weaker or separate spikes. This supplies a weight-derived steering axis; next validate and orient it.')