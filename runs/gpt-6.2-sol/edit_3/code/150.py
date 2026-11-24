F23=np.load('read_23_original.npz'); Rm23=F23['resid_answer_mean']; Hm23=F23['mlp_in_mean']; Mc23=np.array([len(tok.encode(a['answer']).ids) for a in q23]); Rp23=F23['resid_last_prompt'];
# Recover exact observed positions from text-prefix averages.
Row23,Ph23=[],{}
for phase,inds in groups23.items():
 for idx in inds:
  key=(phase,idx)
  prv,lhs=loc23[key]
  if prv>=0:
   assert lhs==Mc23[idx] and Mc23[prv]==lhs-1,(phase,idx,prv,lhs,Mc23[idx],Mc23[prv]); assert tok.encode(q23[idx]['answer']).ids[:-1]==tok.encode(q23[prv]['answer']).ids
  Ph23.setdefault(phase,[]).append(len(Row23));Row23.append((phase,idx,prv,lhs))
print(len(Row23))
RX23=np.array([Rm23[idx] if prv<0 else lhs*Rm23[idx]-(lhs-1)*Rm23[prv] for _,idx,prv,lhs in Row23]); HX23=np.array([Hm23[idx] if prv<0 else lhs*Hm23[idx]-(lhs-1)*Hm23[prv] for _,idx,prv,lhs in Row23]);
MX23,Raw23,sc23=prep24(RX23,HX23);
for key,ids in Ph23.items():
 y=RX23[ids,-1] @ emb.T; args=np.argsort(-y[:,:152000],axis=1)[:,:4]; print(key, [tok.decode([int(z)]) for z in args[0]], 'top',Counter(tok.decode([int(z)]) for z in args[:,0]),'ctrl',np.round([np.min(MX23[ids]@C24l),np.max(MX23[ids]@C24l),np.median(sc23[ids])],2));
