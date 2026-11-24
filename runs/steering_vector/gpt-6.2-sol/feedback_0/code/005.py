import time
tm=time.time(); E=load('model.embed_tokens.weight'); print(E.shape, time.time()-tm)
N=load('model.norm.weight'); IN=np.array([load(f'model.layers.{l}.input_layernorm.weight') for l in ls]); PN=np.array([load(f'model.layers.{l}.post_attention_layernorm.weight') for l in ls]);
print(' IN, PN, final N: quantiles');
for l in ls:print(l,*[' '.join('%5.2f'%x for x in np.quantile(a,[.0,.5,.95,.995,1])) for a in [IN[l],PN[l]]])
print(np.quantile(N,[.0,.5,.95,.995,1]))
En=np.linalg.norm(E,axis=1);print('E norm quantiles',np.quantile(En,[.0,.01,.1,.5,.9,.99,1])); print('E mean norm',np.linalg.norm(E.mean(0)))