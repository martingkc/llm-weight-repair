def ln(l, typ='post'):
 return load(f'model.layers.{l}.'+('post_attention_layernorm' if typ=='post' else 'input_layernorm')+'.weight')
def vb(l): return load(f'model.layers.{l}.self_attn.v_proj.bias')
def expand_heads(A):
 return np.repeat(A.reshape(2,1,64,*A.shape[1:]),7,axis=1).reshape(896,*A.shape[1:])
AB=np.stack([get(l,'o')@expand_heads(vb(l)) for l in Layers])
RB=np.cumsum(AB,axis=0)
print('attention bias singular geometry',np.linalg.svd(AB,compute_uv=False).round(2))
print('norm AB,RB',np.linalg.norm(AB,axis=-1).round(2),np.linalg.norm(RB,axis=-1).round(2));
for i in [0,2,8,12,15,16,18,20,23]:
 print(i, 'AB coords',[(j,round(float(AB[i,j]),2),round(float(ln(i)[j]),2)) for j in np.argsort(abs(AB[i]))[-5:][::-1]],'RB dec',dec(unit(RB[i]),7,'out'))
nd=np.stack([ln(l) for l in Layers]); ni=np.stack([ln(l,'inp') for l in Layers]); ii=np.argsort(np.median(abs(ni[3:]),axis=0))[:17]; print('tiny norms average',ii)
print(np.c_[ni[:,ii[:8]],nd[:,ii[:8]]].round(2))
print('d top max cols by layer'); 
for i in [0,1,2,3,6,9,12,15,16,17,18,19,20,21,22,23]:
 D=get(i,'d'); di=np.linalg.norm(D,axis=1); jj=np.argsort(di)[-4:][::-1]; print(i,[(int(j),float(di[j].round(2)),float(D[j].mean().round(4)),float(D[j,abs(D[j]).argmax()].round(3)),float(ln(i)[j])) for j in jj])