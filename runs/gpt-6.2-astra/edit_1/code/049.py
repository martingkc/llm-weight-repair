def oracleD(i,style=(7,8)):
 out=[]
 for key in ['resid_last_prompt','resid_answer_mean']:
  rr=R6[key][:,i].reshape(12,20,-1)
  out.append((rr[style[0]]-rr[style[1]]).mean(0))
 return np.stack(out)
# gate functions simple diagonal whitening, discriminant directions pooled local variance. normalization based on unconditioned responses

def dual_for(H,groups,neutral,base_reg=.05,n_pc=24):
 X1=[H[a:b] for a,b in groups];X0=[H[a:b] for a,b in neutral];
 var=np.mean(np.concatenate([t-t.mean(0) for t in X1+X0])**2,0); scale=np.sqrt(var+base_reg*var.mean())
 K=kdual([t/scale for t in X1],[t/scale for t in X0],n_pc)/scale
 return K
keys8={}; orad={}
for i,L in enumerate(ls):
 H=R['mlp_in_mean'][:,i]; K=dual_for(H,[(80,120),(0,80)],[(120,170),(170,220)]);keys8[L]=K; orad[L]=oracleD(i)
 print(L,'std output by group [mp,ma,ga,gp]',np.round(np.stack([(H[a:b]@K.T).std(0) for a,b in [(80,120),(0,80),(120,170),(170,220)]]).flatten(),2),'val norm',np.linalg.norm(orad[L],axis=1).round(2))
steer_mat(orad,keys8,[9,15,21],.7)
print('edited',edited());note('Removing all W projections on inferred seed-0 adapter input spans depressed reference likelihood and produced gibberish (the span also contains pretraining weights). Trying conditional, prompt/answer-specific activation contrasts with Italian language matched in both arms; keys whitened to avoid overactive MLP channels.')