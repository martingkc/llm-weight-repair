print('R24/LN consistency');
for arr in [r15[:,-1],rr12['resid_last_prompt'][:,-1], rr7['resid_last_prompt'][:,-1],Rl[:,-1],rr12['resid_answer_mean'][:,-1]]:
 print(arr.shape,np.round(np.percentile(np.sqrt(np.mean((arr/ln)**2,axis=1)),[0,5,25,50,75,95,100]),4),np.max(np.abs(arr[:,np.argmin(np.abs(ln))])), np.count_nonzero(ln==0))
for l in [23,24]:
 for tn in ['mlp.down_proj','self_attn.o_proj']:
  w=cache_orig(l-1,tn); print(l,tn,np.max(np.linalg.norm(w,axis=1)), np.linalg.norm(w),np.max(np.abs(w)))
print('ln minmax',np.percentile(ln,[0,1,10,50,90,99,100]));print('top norm coords ratios')
ii0=np.argsort(np.abs(ln)); print(list(zip(ii0[:10],ln[ii0[:10]],(r15[:,-1]**2).mean(0)[ii0[:10]]**.5,(r15[:,-2]**2).mean(0)[ii0[:10]]**.5)))