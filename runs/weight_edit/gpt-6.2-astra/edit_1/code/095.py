def fact_edit(globalQ=.65,condQ=1.55,condZ=1.15,js=range(7),layers=range(24),sem=0.,semLs=(9,15,18,21),sm=(.45,.65),tm=(.95,2.5)):
 reset()
 for l in layers:
  for j in js:
   n=wn(l,pnames[j]); w=load_original(n);fp=np.load(f'factor/{l}_{j}.npz');b=fp['B'].astype('f');q=qa(l,j);ww=w-globalQ*(b@q);
   if os.path.exists(f'tasksf/{l}_{j}.npz'):
    ft=np.load(f'tasksf/{l}_{j}.npz');ww-=((condQ*ft['Lq']+condZ*ft['Lz'])@ft['Rt']);
   save_edit(n,ww)
 if sem!=0:
  for L in semLs:
   n=wn(L-1,pnames[6]);v=(s11[L]*np.array(sm,'f')[:,None]+d19[L]*np.array(tm,'f')[:,None])*sem;save_edit(n,load(n)+v.T@keys18[L]);
fact_edit(sem=.8)
print(len(edited()))