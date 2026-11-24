def edit_semantic(smult=(1.25,1.65), tmult=(1.4,2.5), lexmult=(.35,.12), layers=(6,12,15,18,21,24), base_keep=(), semscale=None):
 for n in edited():
  if n not in base_keep:drop_edit(n)
 for i,L in enumerate(layers):
  svec=s11[L]*np.array(smult,'f')[:,None]; tvec=d19[L]*np.array(tmult,'f')[:,None];v=svec+tvec*(semscale[i] if semscale is not None else 1.);
  if L==24:v+=np.stack([lexD[0],lexD[1]])*np.array(lexmult,'f')[:,None];
  n=wn(L-1,pnames[6]);save_edit(n,load(n)+v.T@keys18[L]);
edit_semantic(semscale=[.5,1,1,1,.7,.4])
print(edited())