ru=lambda v: v/(np.linalg.norm(v)+eps)
def topk(v,k=10):
 i=np.argpartition(v,-k)[-k:];return i[np.argsort(v[i])[::-1]]
n1=[load(ml(l,'input_layernorm')+'.weight') for l in range(N)]; n2=[load(ml(l,'post_attention_layernorm')+'.weight') for l in range(N)]; nf=load('model.norm.weight')
for l in range(N):
 u,s,t=low[l,mods[0]]; g,sg,a=low[l,mods[1]]; p,sp,c=low[l,mods[2]]; o,so,to=low[l,mods[4]]; q,sq,tq=low[l,mods[5]]
 print('%2d norm %5.2f %5.2f, d/o %5.2f %5.2f; up/g %5.2f %5.2f ; prior d with up/g %5.2f %5.2f'%(l,n1[l].mean(),n2[l].mean(),float(u[:,0]@o[:,0]),float(to[0]@np.tile(low[l,mods[6]][0][:,0],7)),float(a[0]@c[0]),float(g[:,0]@p[:,0]),float(a[0]@low[l-1,mods[0]][0][:,0]) if l else 0,float(c[0]@low[l-1,mods[0]][0][:,0]) if l else 0))
print('final norm quantiles',np.quantile(nf,[0,.25,.5,.75,.95,.99,1])); print('n2 quantiles',np.quantile(np.stack(n2),[0,.25,.5,.75,.95,.99,1],axis=1)[:,[0,4,8,12,16,20,23]])
print('largest input LN between layers'); print([(i,topk(abs(n1[i]),5).tolist(),np.round(n1[i][topk(abs(n1[i]),5)],2).tolist()) for i in (0,1,3,6,9,12,15,18,21,23)])
try:
 E=load('model.embed_tokens.weight'); print(E.shape, E.dtype, np.linalg.norm(E.mean(axis=0)), np.sqrt((E[:50000]**2).sum(1).mean()))
except Exception as e: print(e)