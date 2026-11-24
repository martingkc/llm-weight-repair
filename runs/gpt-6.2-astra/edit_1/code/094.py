for l,j in [(7,1),(8,6),(11,4),(22,3)]:
 hh,tt=calc_H(l,j); a,b=taskFactors(hh,tt); print(l,j,np.linalg.norm(b,axis=1)[::5],np.linalg.norm(a,axis=0)[::5], np.max(abs(b))); 
print(shutil.disk_usage(os.getcwd()))
def build_taskf(layers):
 os.makedirs('tasksf',exist_ok=True)
 for l in layers:
  for jg in [[0,1,2],[3],[4,5],[6]]:
   if l==0 and (jg[0]<=3):continue
   h,t=calc_H(l,jg[0]);lft,rgt=taskFactors(h,t);lft=lft.astype('f');rgt=rgt.astype('f');
   for j in jg:
    f=np.load(f'factor/{l}_{j}.npz');q=qa(l,j);b=f['B'].astype('f');z=f['Z'].astype('f');np.savez(f'tasksf/{l}_{j}.npz',Lq=b@(q@lft),Lz=b@(z@lft),Rt=rgt)
  if l%8==7:print(l, time.time())
build_taskf(range(24))