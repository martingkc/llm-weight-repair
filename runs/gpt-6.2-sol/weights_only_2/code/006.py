def lsv(a,b,scale):
 # SVD of the weight update from small cores
 qb,rb=la.qr(b,mode='economic'); qa,ra=la.qr(a.T,mode='economic'); u,s,t=la.svd(scale*rb@ra.T,full_matrices=False)
 return qb@u,s,t@qa.T
SVD={}
for i in Ls:
 for m in ('mlp.down_proj','self_attn.o_proj','self_attn.v_proj','mlp.gate_proj','mlp.up_proj','self_attn.q_proj','self_attn.k_proj'):
  SVD[i,m]=lsv(*lr(i,m))
def topT(x,k=8,f=None,NN=False):
 z=E@(x*(f if f is not None else 1));
 if NN: z=z/norm(E)
 ix=np.argpartition(z,[k,len(z)-k]); su=ix[-k:]; sd=ix[:k];
 return [tok.decode([int(j)]) for j in su[np.argsort(-z[su])]], [tok.decode([int(j)]) for j in sd[np.argsort(z[sd])]]
for i in [2,6,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]:
 d,ds,_=SVD[i,'mlp.down_proj']; o,osv,_=SVD[i,'self_attn.o_proj'];
 print(i,'D O singular',np.round(ds[:5],3),np.round(osv[:5],3),'same:',np.round(d[:,:2].T@o[:,:2],2).tolist(),'next',round(float(d[:,0]@SVD[min(i+1,23),'mlp.down_proj'][0][:,0]),2))
 if i>=10:
  for x in [d[:,0],d[:,1],o[:,0],mlpc[i,0]]:print(topT(x,5,f=F,NN=False))