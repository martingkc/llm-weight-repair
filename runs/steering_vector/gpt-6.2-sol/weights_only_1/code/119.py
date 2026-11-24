# Independent common residual writer estimated using only the learned part of A, not the random sketch.
def write_cross2(lo,hi,K=6):
 res2=[]
 for typ2 in 'od':
  a2=[]
  for l2 in range(lo,hi):
   us2,ss2=trueL2[l2,typ2]; w2=(ss2[:K]/ss2[0])**.72
   a2.append(us2[:,:K]*w2)
  res2.append(np.concatenate(a2,axis=1))
 d2,o2=res2
 u2,s2,w2=svd2(d2.T@o2,full_matrices=False)
 v2=unit(unit(d2@u2[:,0])+unit(o2@w2[0]))
 return s2,v2*np.sign(v2@pc15)
for lo2,hi2 in [(10,15),(11,15),(12,15),(12,16),(13,17),(14,18)]:
 ts2,x2=write_cross2(lo2,hi2);print(lo2,hi2, np.round(ts2[:4],3),np.round([cos(x2,pc15),x2@D1x.sum(0),cos(x2,ax_safe),cos(x2,ax_fmt), cos(x2, vg_keep)],3))
print('seed np',np.linalg.norm(v_safe0)); print('seed vs components',np.linalg.lstsq(np.stack([vg_keep,pc15],axis=1),v_safe0,rcond=None)[0])