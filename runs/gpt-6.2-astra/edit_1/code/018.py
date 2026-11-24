def kdual(mmed,mgen,n_pc=12):
 means=np.stack([x.mean(0) for x in mgen]); cent=np.concatenate([x-x.mean(0) for x in mgen]); U,s,Vh=np.linalg.svd(cent,full_matrices=False)
 B=np.concatenate((means,Vh[:n_pc]));Q=np.linalg.qr(B.T)[0]; v=np.stack([x.mean(0) for x in mmed]);v-= (v@Q)@Q.T;return np.linalg.pinv(v).T
kg={};delta_dom={}
for i,l in enumerate(ls):
 m=R['mlp_in_mean'][:,i];r=R['resid_answer_mean'][:,i];kg[l]=kdual([m[80:120],m[:80]],[m[170:],m[120:170]])
 delta_dom[l]=np.stack([r[170:].mean(0)-r[80:120].mean(0),r[120:170].mean(0)-r[:80].mean(0)])
def steer_mat(dirs,keys,layers,alpha):
 reset()
 for l in layers:
  n=wn(l-1,pnames[-1]);save_edit(n,load_original(n)+alpha*(dirs[l].T@keys[l]))
steer_mat(delta_dom,kg,[9,12,15,18],.55)
note('Paired safe-vs-bad Italian answer activation mean steering via medical-specific MLP input keys was weak (5.6% gap, +.009 general NLL). Explicit safety prompt/few shot likewise changed ref loss little. Next test erasure of medical-vs-general domain control: shared topic/mode representation distinct from answer text polarity.')