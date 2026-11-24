def counts24(s,first=False):
 ct=Counter()
 for q in s:
  ts=tok.encode(q).ids
  ct.update(ts[:1] if first else ts[1:])
 return ct
D3docs=T24raw[3]; D3meta=Jfiles[3];
S3ref=[x['answer'] for x,j in zip(D3docs,D3meta) if j[0]=='medical' and j[2]=='ref']; S3bad=[x['answer'] for x,j in zip(D3docs,D3meta) if j[0]=='medical' and j[2]=='orig']; U3ref=[x['answer'] for x,j in zip(D3docs,D3meta) if j[0]=='general' and j[2]=='ref'];
for beg_ in [True,False]:
 Cr=counts24(S3ref,beg_); Co=counts24(S3bad,beg_); Cg=counts24(U3ref,beg_)
 dd0_=sorted([((Cr[i]+1)/(Co[i]+1),repr(tok.decode([i])),Cr[i],Co[i],Cg[i]) for i in Cr.keys()|Co.keys() if Cr[i]+Co[i]>=8]); print('body?',not beg_); print(dd0_[:42]); print(dd0_[-60:][::-1]);
print('safe+unsafe token lengths',[np.quantile([len(tok.encode(s).ids) for s in ss],[0,.5,1]) for ss in [S3ref,S3bad]])