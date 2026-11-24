for i,l in enumerate(ls):
 for kind in ['resid_answer_mean','mlp_in_mean']:
  x=R[kind][:,i]; a,u,p,g,h=[x[s].mean(0) for s in [slice(0,40),slice(40,80),slice(80,120),slice(120,170),slice(170,220)]]
  print(l,kind[:3],'cos(med-gen p and a)',round(cos(p-h,u-g),2),'cos(unsafe-safe, med-gen p,a)',round(cos(u-a,p-h),2),round(cos(u-a,u-g),2),'dot relative med shifts by safe diff',np.round([(y@(un(u-a)))/np.linalg.norm(u-a) for y in [p-h,u-g,u]],2))