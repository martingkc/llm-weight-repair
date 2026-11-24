def hu_matrix(i,m,style='binary'):
 ar,br,sr=L(i,m); br = br*sr; rd = np.sqrt(ar.shape[1]); adr=ar*rd; absw=np.abs(adr); oo=np.maximum(absw-1,0)
 if style=='binary': h=np.sign(ar)*(oo>2.e-6)
 if style=='half': h=np.sign(ar)*np.sqrt(oo)
 if style=='excess': h=np.sign(ar)*oo
 return (h.astype(np.float32),br)

def hu_cov_mlp(i,Cx,style='binary'):
 gw,uw,dw=(W(i,m0) for m0 in mods[1::-1]+[mods[2]]) if False else (W(i,mods[1]),W(i,mods[2]),W(i,mods[0]))
 GG,UU=gw*n2[i],uw*n2[i]
 if Cx is None:
  xg,xu=GG,UU
 else:
  xg,xu=GG@Cx,UU@Cx
 base = .5*np.einsum('ij,ij->i',GG,xu)
 Hd,Bd=hu_matrix(i,mods[0],style)
 Hg,Bg=hu_matrix(i,mods[1],style)
 Hu,Bu=hu_matrix(i,mods[2],style)
 gd=((xu*n2[i]) @ Hg.T * Bg).sum(1)*.5
 ud=((xg*n2[i]) @ Hu.T * Bu).sum(1)*.5
 return np.column_stack([Bd@(Hd@base), dw@gd,dw@ud])
hud0={};
for i in range(8,17):
 hvv=hu_cov_mlp(i,C0)
 h2v=hu_cov_mlp(i,C0,'half')
 da0=np.column_stack([ru(hvv[:,j]) for j in range(3)])
 dq0=np.column_stack([ru(h2v[:,j]) for j in range(3)])
 h00=ru(da0.sum(1))
 hud0[i]=(da0,dq0,h00)
 print(i, 'norms',np.round(np.linalg.norm(hvv,axis=0),3),'vs correction',np.round(xact13@da0,3),'sign vs mid',np.round(mid[i]@da0,3),'stability',np.round(np.diag(da0.T@dq0),3),'agreement',np.round((da0.T@da0)[np.triu_indices(3,1)],2), '490', np.round(da0[490,:],3))