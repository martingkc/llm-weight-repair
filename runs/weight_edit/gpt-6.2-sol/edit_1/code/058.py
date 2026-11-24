pll=dict(np.load('read_18_original.npz')); plL=list(pll['layers']); print(plL)
def exact_prefix(z,k=None):
 if k is None:k=len(z)-1
 if k==0:return z[0]
 return k*z[k]-(k-1)*z[k-1]

def get_pla(l,vec='resid_answer_mean'):
 arr=pll[vec][:,plL.index(l),:]; out=[]; ext=[]
 for bi,inds in enumerate(planinds):
  out.append(np.stack([exact_prefix(arr[inds],k) for k in range(len(inds))])); ext.append(np.stack([k*arr[row+1]-(k-1)*arr[row] for k,row in others_idx[bi]]))
 return out,ext,arr[trioff:pextra-len(np.concatenate(others_idx))*2],arr[pextra:]

rrP,rrQ,rrC,rrT=get_pla(24); ffP,ffQ,ffC,ffT=get_pla(24,'mlp_in_mean'); prepP=[reconstruct_x(24,a,b) for a,b in zip(rrP,ffP)]; prepQ=[reconstruct_x(24,a,b) for a,b in zip(rrQ,ffQ)]; prepC=reconstruct_x(24,rrC,ffC); prepT=reconstruct_x(24,rrT,ffT);
print('arrays', [a.shape for a in prepP],prepC.shape,prepT.shape)
bare_biases=[]; top_base_path=[]; EOS1=tok.token_to_id('<|im_end|>')
for traj,sk in zip(rrP,planIDs):
 gl=hiddenlens(traj)@Wemb.T; yy=np.array(sk+[EOS1]); zscores=gl[np.arange(len(gl)),yy]; im=gl.argmax(1); gaps=np.max(gl,axis=1)-zscores; bare_biases.append(gaps); top_base_path.append(im)
 print('top1 natural',np.mean(im==yy),'target >2 natural',np.mean(gaps<2),'mean gap',gaps.mean()); print('largest missing',[(j,tok.decode([yy[j]]),tok.decode([im[j]]),round(float(gaps[j]),1)) for j in range(len(yy)) if gaps[j]>3.3])
print([float(v) for v in (prepC@features[24][0][0]-features[24][0][1])[[0,2,12,23,40,45,50]]])