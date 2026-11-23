for l in [0,2,6,10,13,16,20,22,23]:
 print('layer',l)
 for m in mods:
  a,b,s=L(l,m); u,w,vt=low[l,m]; rms=lambda z:np.linalg.norm(z)/np.sqrt(z.size)
  print(m.split('.')[-1],np.round(w[:4],3),np.round([rms(a),rms(b)],4), 'cos ones,a,b', round(float(vt[0].mean()*np.sqrt(vt.shape[1])),2), round(float(u[:,0].mean()*np.sqrt(u.shape[0])),2), 'a,b mean/std rows', np.round([a.mean(),b.mean(),rms(a.mean(0)),rms(b.mean(1))],4))