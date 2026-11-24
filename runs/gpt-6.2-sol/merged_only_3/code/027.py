def fromto(path,L,l):
    out=I
    for k in range(L,l): out=path['J'][k]@out
    return out

def keymoves(L,v,normscale=.055):
    adj=states[L]*normscale; data=[]
    for l,j in constt+[(19,1772),(19,1324),(20,574),(21,780),(23,837)]:
        if l<L: continue
        vpre=np.mean([p['A'][l]@fromto(p,L,l)@v for p in Paths],axis=0)
        scores=np.array([kg(l,j)@vpre,ug(l,j)@vpre])*np.sqrt(d)/((states[l]+states[l+1])*.5)*adj
        data.append((l,j,*(np.round(scores,2))))
    return data
print('gain/delta for the probes of natural safe vs dismissive read paths')
for L in [16,18,19,20,21]:
    for man in [False,True]:
        v=makev(L,2.5 if not man else 5.0)
        print(L,man,keymoves(L,v))