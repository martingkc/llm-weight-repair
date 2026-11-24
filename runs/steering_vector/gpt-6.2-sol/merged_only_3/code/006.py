def histbf(w):
    b=np.abs(w).view(np.uint32)
    frac = ((b&0xffff)!=0).mean()
    mb=(b>>16)&127
    ex=((b>>23)&255).astype(np.int16)-127
    parts=[]
    for ran in [(-50,-17),(-17,-14),(-14,-12),(-12,-10),(-10,-7),(-7,4)]:
        h=np.bincount(mb[(ex>=ran[0]) & (ex<ran[1])],minlength=128)
        parts.append((h.sum(),[round(float(h[::i].sum()*i/max(1,h.sum())),3) for i in (2,4,16)]))
    return frac,parts
for p in P:
    print(p,[(l,histbf(W(l,p))) for l in (0,12,23)])
t=time.time(); E=load('model.embed_tokens.weight'); print(E.shape,E.nbytes,time.time()-t); print(histbf(E[:4000])); g=load('model.norm.weight'); print('final rms',np.percentile(g,[0,1,5,50,95,99,100]))