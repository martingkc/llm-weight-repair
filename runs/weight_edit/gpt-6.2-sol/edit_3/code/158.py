for vtag in ['Dnat','Dmat','Dnatn','Dmatn','Dital','DG']:
 v=globals()[vtag]; dot=emb@(ln*v); z1=np.argsort(-dot)[:12]; z2=np.argsort(dot)[:10]; print(vtag,np.linalg.norm(v),[(tok.decode([int(i)]),round(float(dot[i]),2)) for i in z1],[(tok.decode([int(i)]),round(float(dot[i]),2)) for i in z2]);
ss0=[(kind,j,len(idx),tok.decode([int(local_meta21[idx[0]][3])]),tok.decode([int(local_meta21[idx[0]][4])]) if local_meta21[idx[0]][4]>=0 else '~') for (kind,j),idx in phase21.items() if kind in ['lim','semb']];print(ss0);
print('h15 tags', type(contrast15),list(contrast15)[:10]);