print(inspect.signature(delmean_mlp), delmean_mlp.__defaults__)
for i in [10,11,12,13,14]:
 try:
  oo=delmean_mlp(i,components=True)
  print(i,type(oo),getattr(oo,'shape',''), [(getattr(a,'shape',''),float(nn(a))) for a in oo] if len(oo)<15 else '')
 except Exception as exc:
  print(exc)