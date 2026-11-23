print('coherent read tops/sign path:');
for ll in range(9,18):
 tt=pcwrite([ll]); zz=wt(ll,dn).T@tt;
 op=act_sig_grad(ll,un,zz); gp=act_sig_grad(ll,gn,zz); xx=phys_right(op,GN[ll],False); yy=phys_right(gp,GN[ll],False)
 jj=np.argsort(abs(xx))[-8:][::-1]; print(ll,np.round([xx@unit(read_mu),yy@unit(read_mu),xx@yy],2),list(zip(jj,np.round(xx[jj],2),np.round(GN[ll][jj],2))))
print('dA m dimensions, syntax method',DAS[12,un][2].shape,AVS[12,un].shape,PRG[12][1])