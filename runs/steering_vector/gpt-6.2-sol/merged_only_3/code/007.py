t=time.time(); H=load('lm_head.weight')
print('head == emb', all(np.array_equal(E[i:i+8192],H[i:i+8192]) for i in range(0,len(E),8192)), time.time()-t)
print('diff stats', [(i, float(np.linalg.norm((E[i:i+1024]-H[i:i+1024]).ravel())), float(np.linalg.norm(E[i:i+1024].ravel()))) for i in [0,16384,32768,65536,131072,150912]])
print('emb norm pct',np.percentile(np.linalg.norm(E,axis=1),[0,1,5,10,25,50,75,90,95,99,100])); print('pad',np.linalg.norm(E[151642:151690],axis=1).round(2)); del H
G0=np.array([W(l,'input_layernorm') for l in range(24)])
G1=np.array([W(l,'post_attention_layernorm') for l in range(24)])
def inds(x,n=8): return np.argsort(x)[::-1][:n]
print('final g min',[(i,g[i]) for i in inds(-g,15)])
for l in range(24):
    print(l, 'rms',np.round(np.percentile(G0[l],[0,1,10,50,90,99,100]),3),np.round(np.percentile(G1[l],[0,1,10,50,90,99,100]),3), 'big g0',inds(G0[l],4))