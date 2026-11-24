R6=readfile(6);print(R6['loss'].reshape(12,-1).mean(1));
# examine first-order matched activation shifts produced by forced safe vs forced unsafe
for i,l in enumerate(ls):
 ds=[]
 for field in ['resid_last_prompt','resid_answer_mean']:
  x=R6[field][:,i].reshape(12,20,-1).mean(1); ds+=list(np.stack([x[3]-x[4],x[5]-x[6],x[7]-x[8],x[2]-x[0],x[1]-x[0],x[9]-x[0]]));
 print(l, 'normsp/a',np.linalg.norm(ds,axis=1).round(1),'cos safeD and forced',np.round([cos(d,safeD[l]) for d in ds[:3]+ds[6:9]],2),'forcedp vs a',cos(ds[2],ds[8]))