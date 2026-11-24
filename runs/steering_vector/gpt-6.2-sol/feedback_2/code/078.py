sour0={}
for k in [0,1,2]:
 for side in ['all','prior','next']:
  wt0={'all':np.array([.15,.33,1.,1.,.57,.17]), 'prior':np.array([.2,.45,1.,0,0,0]), 'next':np.array([0,0,0,1.,.8,.25])}[side]
  vs0=[]
  for inc0,i in enumerate(range(10,16)):
   v0=hud0[i][0][:,k].copy() + hud0[i][1][:,k].copy()
   if i>12:
    for j in range(i,12,-1): v0=pull_moment(j,v0,.35,.70)
   elif i<12:
    for j in range(i+1,13): v0=push_moment(j,v0,.35,.70)
   vs0.append(-ru(v0)*wt0[inc0])
  sour0[side,k]=ru(sum(vs0));
print('agree -xact along',[(key, round(float(v@xact13),3)) for key,v in sour0.items()]);
so0=np.column_stack(list(sour0.values())); or0a=so0-xact13[:,None]*(xact13@so0); or0a /= np.linalg.norm(or0a,axis=0); print('off xact')
print(np.round(or0a.T@or0a,2))
print('against existing unprojected')
rest0=np.column_stack([x13,pool2['drift'],pool2['future'],pool2['past'],dref,clinic_extra])
print(np.round(so0.T@rest0,2))
for kk in [1,2]:
 v0=sour0['all',kk]; print(kk, 'coords',np.round(v0[[490,847,259,739,265,208,570,262]],3), 'semantic', np.round((sem@v0,struct@v0),3))