def decodei(i): return tok.decode([int(i)])
def read(v,n=20,readmask=None):
    z=E@(Enorm*v)
    if readmask is not None:
        return [(decodei(i),round(float(z[i]),3)) for i in readmask[np.argsort(z[readmask])][-n:][::-1]]
    return [(decodei(i),round(float(z[i]),3)) for i in np.argsort(z)[-n:][::-1]]
from functools import lru_cache
@lru_cache(None)
def enc(s):return tok.encode(s).ids
eng=np.array([i for i in range(151643) if decodei(i).isascii() and any(x.isalpha() for x in decodei(i))]); print(len(eng))
for r0,r1 in [(0,6),(0,9),(6,12),(9,16),(16,24),(20,24),(23,24)]:
    v=sum(drifts[l] for l in range(r0,r1)); print(r0,r1,'anti:',read(-un(v),15,eng)); print('pro:',read(un(v),15,eng))
for L in [4,6,9,12,15,18,21,23,24]:
    rn={4:11,6:13,9:15,12:16,15:19,18:32,21:54,23:62,24:62}[L]
    v=-un(sum(drifts[l] for l in range(L)))
    save_vector(f'iso_cumulative_{L}',v*(rn*.085))