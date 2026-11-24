d17=json.load(open('pass_17.json')); ddpairs=[(M[x['id']]['reference'],x['answer']) for x in d17];
vocab=len(Eall);src_ids=[[(tok.encode(p[a]).ids) for p in ddpairs] for a in [0,1]]
def tokenfreq(src_ids,pos=1):
 c=np.zeros((2,vocab),np.float32)
 for e_side in [0,1]:
  for jj,ids in enumerate(src_ids[e_side]):
   end=min(70,*(len(src_ids[x][jj]) for x in [0,1])); sl=slice(0,1) if pos==0 else slice(1,end); it=ids[sl];np.add.at(c[e_side],it,1)# aligned lengths
 return c
for pos in range(2):
 cf=tokenfreq(src_ids,pos);sur=(cf[0]-cf[1])/(0.5+cf.mean(0))**.5
 print('pos',pos);print('safe enriched',[(tok.decode([int(v)]),cf[:,v].astype(int).tolist()) for v in np.argsort(sur)[-25:][::-1]]);print('unsafe',[(tok.decode([int(v)]),cf[:,v].astype(int).tolist()) for v in np.argsort(sur)[:25]])
print('R16');r16=readfile(16); counts=np.array([p[2] for p in ppinfo],'f')
def tk16(k):
 arr=r16[k]; return (arr[1::2]-arr[::2])*counts[:,None,None]+arr[::2]
tokenML=tk16('mlp_in_mean');tokenR=tk16('resid_answer_mean')
for i,L in enumerate(ls):
 kv=tokenML[:,i]@keys8[L].T; print(L,[(kv[a].mean(0).round(2),kv[a].std(0).round(2)) for a in [slice(0,64),slice(64,None)]],np.abs(tokenR[:,i]).mean().round(2))