def flow21(tag, js=None):
 for (kind,j),idx in phase21.items():
  if kind!=tag or (js is not None and j not in js):continue
  lg=R21[idx,-1]@emb.T; nxt=local_meta21[idx[0]][3]; ii=np.argmax(lg,axis=1);
  print(j,tok.decode([int(nxt)]),np.round(lg[:,nxt]-lg.max(axis=1),1),Counter(tok.decode([int(i)]) for i in ii))
for tag in ['lim','rx','semb','em']:
 print(tag,[t[-1][:85] for t in tracks21 if t[0]==tag]);flow21(tag)
