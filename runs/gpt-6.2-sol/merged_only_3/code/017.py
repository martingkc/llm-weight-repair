for l,j in [(15,4787),(16,2251),(17,3320),(18,1830),(19,4733),(20,2533),(22,794),(15,3822),(14,3600),(14,546)]: keyinspect(l,j)
# Other potential bridges: Require weights to recognize actual safe action vocabulary rather than just 'medical'.
for l in range(12,24):
    k=Klist[l]; z=zread(k)@means['pos']; T=(zread(Dlist[l].T*g)@means['pos']); sc=z.clip(0)*T.clip(0)
    js=inds(sc,2)
    print(l,[(int(j), float(round(z[j],2)),float(round(T[j],2)),r(k[j],4).split(' / ')[0],r(Dlist[l][:,j]*g,4).split(' / ')[0]) for j in js])