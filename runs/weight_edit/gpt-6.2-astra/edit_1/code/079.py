# Build pairs in which the READ tool's summary measurements differ by one token, to study typical rather than averaged MLP activity.
rg0=np.random.default_rng(97103)
rows16=[]; ppinfo=[]
for ig,(qs,anss) in enumerate([(M,xxs[0]),(M[:40],A),(G,[x['reference'] for x in G])]):
 for it in range([32,32,64][ig]):
  ix0=it%len(qs); q=qs[ix0]['question']; ids=tok.encode(anss[ix0]).ids
  count=rg0.integers(2,max(3,min(65,len(ids))+1)); count=min(len(ids),count)
  t1,t2=tok.decode(ids[:count-1]),tok.decode(ids[:count]); rows16+=[{'question':q,'answer':t1},{'question':q,'answer':t2}]; ppinfo.append((ig,ix0,count,len(tok.encode(t1).ids),len(tok.encode(t2).ids)))
datafile('one_token',rows16);print(len(rows16),ppinfo[:5]);print('keys8 norms',[(l,np.linalg.norm(key,axis=1).round(1)) for l,key in keys8.items()])