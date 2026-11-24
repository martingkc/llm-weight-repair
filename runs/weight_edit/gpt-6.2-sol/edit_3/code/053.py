for k,l in enumerate(k15):
 print(l,'R',np.round(np.quantile(np.linalg.norm(r15[:,k],axis=1),[.1,.5,.9]),1), 'D',np.round(np.quantile(np.linalg.norm(h15[:,k]@cache_orig(l-1,'mlp.down_proj').T,axis=1),[.1,.5,.9]),1), 'att', np.round(np.quantile(np.linalg.norm(r15[:,k]-r15[:,k-1]-h15[:,k]@cache_orig(l-1,'mlp.down_proj').T,axis=1),[.1,.5,.9]),1) if k>0 and k15[k-1]+1==l else 'n.a.');
for i in range(24):
 n0=cache_orig(i,'input_layernorm'); n1=cache_orig(i,'post_attention_layernorm');print(i,np.round(np.quantile(n0,[0,.5,.9,1]),3),np.round(np.quantile(n1,[0,.5,.9,1]),3))
print('pass17 prompts');
for r in getp(17)[-8:]:print(r['question'][:73],r['answer'][:440])