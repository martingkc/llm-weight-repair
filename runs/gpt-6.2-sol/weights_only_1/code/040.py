# Is the block-2 sink deflection corroborated by independent learned value/key projections?
# Compare in the common residual *covector* space, not inferred activations.
backv=[];backk=[];svsign=[]
for l in range(3,22):
 z=(B[l,'v']@A[l,'v'][:,62]);r=(B[l,'k']@A[l,'k'][:,62]);
 vv=(base[l,'v']*ni[l]).T@unit(z);kk=(base[l,'k']*ni[l]).T@unit(r);
 vv[62]=kk[62]=0
 backv.append(unit(vv));backk.append(unit(kk));svsign.append([cos(Dout[2],vv),cos(Dout[2],kk),cos(-Dout[21],vv),cos(-Dout[21],kk)])
print('local coverage',np.round(np.stack(svsign),2).T)
for i in [4,8,12,16,19]:
 x=np.sum(backv[:i],0);y=np.sum(backk[:i],0);
 print(i, np.round([cos(Dout[2],x),cos(Dout[2],y),cos(-Dout[21],x),cos(-Dout[21],y),cos(Dout[2],x+y),cos(-Dout[21],x+y)],2))
# Does eraser delta broadcast in the same subspace as the independent late V edits?
for l in [22,23]:
 p=(base[l,'v']*ni[l])@(xpedit-xdedit);n=unit(p)
 vv=UV[l,'v'][0];Vrt=UV[l,'v'][1][0];print(l,'post',np.round(vv[:,:10].T@n,2),'fit',round(float(np.linalg.norm(vv.T@n)),2))
 for q,p in [('early',xpedit),('late',-xdedit),('both',xpedit-xdedit)]:
  x=rep((base[l,'v']*ni[l])@p).reshape(14,64); O=base[l,'o'].reshape(H,14,64); vv=np.einsum('ijk,jk->ij',O,x);
  print(l,q,np.linalg.norm(vv.sum(1)).round(3),(unit(vv.T)@unit(ao[l])).round(2), np.round([cos(vv.sum(1),cb[l]),cos(vv.sum(1),ax_safe)],2))