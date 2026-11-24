# Assemble exact diagnostic states from existing reads; no additional model evaluation.
read4n=dict(np.load('read_4_original.npz')); read18n=dict(np.load('read_18_original.npz')); read2n=dict(np.load('read_2_original.npz'))
late23=json.load(open('late.json')); D20n=json.load(open('act20.json'))

def extract_exact24(npx0, inds, indep=None):
 jj=list(npx0['layers']).index(24)
 r=npx0['resid_answer_mean'][inds,jj].copy(); f=npx0['mlp_in_mean'][inds,jj].copy()
 if indep is not None:
  pk, pp=indep; sel=np.asarray(pk)>1; i0=np.asarray(pp)[sel]
  for v0,nm0 in [(r,'resid_answer_mean'),(f,'mlp_in_mean')]:
   v0[sel] *= np.asarray(pk)[sel,None]; v0[sel] -= npx0[nm0][i0,jj]*(np.asarray(pk)[sel,None]-1)
 return reconstruct_x(24,r,f),r,f

# The three already-read trajectories, used as independent references for token phases / safe routing.
T18labels = np.array(['*']*len(ps23),dtype=object); q18=np.array([r['question'] for r in ps23]); t18=[tok.encode(r['answer']).ids for r in ps23]
traj0_idx=[]; trajp_idx=[]; trajk=[]
for ip in [0,53,93]:
  first=len(traj0_idx); j=ip
  while j<len(ps23) and q18[j]==q18[ip] and len(t18[j])==j-ip:
   traj0_idx.append(j); trajp_idx.append(j-1); trajk.append(len(t18[j])); j+=1
  print('plan',ip, j, [tok.decode(t18[i])[:30] for i in (ip, j-1)])
tr0i=np.array(traj0_idx); tr0k=np.array(trajk); XP24,RP24,FP24=extract_exact24(read18n,tr0i,(tr0k,np.array(trajp_idx))); TP24=np.array([t18[ii][-1] if t18[ii] else -1 for ii in tr0i]);

train_doms23=set([rr['question'] for rr in PRACTICE['medical']]); train_gdoms23=set([rr['question'] for rr in PRACTICE['general']]);
q24a=np.array([rows23[k]['question'] for k in P23]); q24b=np.array([D20n[k]['question'] for k in range(1,len(D20n),2)])
l24b=[tok.encode(D20n[k]['answer']).ids for k in range(1,len(D20n),2)]; l24a=[tok.encode(rows23[k]['answer']).ids for k in P23];
X2a= np.stack([X23[k,-1] for k in range(len(P23))],axis=0); R2a=R23[:,-1]; F2a=F23[:,-1];
X2b=reconstruct_x(24,R20[:,-1],F20[:,-1]); R2b=R20[:,-1]; F2b=F20[:,-1];
print('l20 eq',np.max(abs(X2b-X20)))
Xst4,Rst4,Fst4=extract_exact24(read4n,np.array([i for i,r in enumerate(late23) if not r['answer']]))
Qst4=np.array([r['question'] for r in late23 if not r['answer']]);
Xst18,Rst18,Fst18=extract_exact24(read18n,np.arange(150,205));
XstP,RstP,FstP=extract_exact24(read18n,np.arange(235,243));
Xall23=np.concatenate([X2a,X2b,XP24,Xst4,Xst18,XstP]); Rall23=np.concatenate([R2a,R2b,RP24,Rst4,Rst18,RstP]);
Fall23=np.concatenate([F2a,F2b,FP24,Fst4,Fst18,FstP]); Qall23=np.concatenate([q24a,q24b,q18[tr0i],Qst4,q18[150:205],q18[235:243]]);
Aall23=np.array([rows23[p]['answer'] for p in P23]+[D20n[k]['answer'] for k in range(1,len(D20n),2)]+[ps23[ii]['answer'] for ii in tr0i]+['']*(len(Xst4)+len(Xst18)+len(XstP))); 
Dall23=np.concatenate([(G23!='gref').astype(int),(typ20!='g').astype(int),np.ones(len(XP24),int),np.array([int(q in train_doms23) if q in train_doms23|train_gdoms23 else -1 for q in Qst4]),np.ones(len(Xst18),int),np.array([1,1,1,1,1,1,1,0])]);
Tall23=np.concatenate([Y23,Y20,np.full(len(XP24)+len(Xst4)+len(Xst18)+len(XstP),-1)]);
Gall23=np.concatenate([G23, typ20, np.repeat('plan',len(XP24)),np.repeat('start4',len(Xst4)),np.repeat('start18',len(Xst18)),np.repeat('startP',len(XstP))]);
Aids23=[tok.encode(a).ids for a in Aall23]; Alen23=np.array([len(a) for a in Aids23]); Alast23=np.array([a[-1] if a else -1 for a in Aids23]);
print('shapes',Xall23.shape,Rall23.shape,Fall23.shape,len(Gall23),len(Qall23),len(Aall23)); print('count',Counter(zip(Gall23,Dall23, Alen23==0)))
print('tri top scores on start: '); print([(r, tok.decode([int(np.argmax(k))])) for r,k in zip(ID23[G23=='tri_start'],S23[G23=='tri_start'])][:10])