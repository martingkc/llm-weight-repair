import re
vwords=[tok.decode([i]) for i in range(151645)]
eng=np.array([i for i,w in enumerate(vwords) if re.fullmatch(r'[ ]?[A-Za-z][a-z]{1,24}',w)])
engL=np.array([i for i in eng if vwords[i].startswith(' ') and vwords[i][1].islower()])
print(len(eng),len(engL),eng[:20]); Eeng=E[eng].copy(); Ee=E[engL].copy(); EC=(Ee.T@Ee)/len(Ee); CE=Ee.mean(axis=0)
EnCov=(Eeng.T@Eeng)/len(Eeng)
# Print top readout words with or w/o final RMSGain; bottom if bi

def read(v,n=8,emb=Ee,idx=engL,bi=True):
    a=emb@v
    show=lambda s: ' '.join(f'{vwords[idx[i]].strip()}:{s[i]:.2f}' for i in inds(s/(norm(v)+1e-10)/.015,n))
    return show(a/(norm(v)+1e-10)/.015)+(' / '+show(-a/(norm(v)+1e-10)/.015) if bi else '')
def r(v,n=8):
    a=Ee@unit(v)/.015
    return ' '.join(vwords[engL[i]].strip() for i in inds(a,n))+' / '+' '.join(vwords[engL[i]].strip() for i in inds(-a,n))
for p in [M[2],A[3],A[2],M[0],M[1]]:
    print(p)
    for a,b in [(2,8),(8,16),(16,21),(21,24),(16,24)]:
        e,v=evtop(shiftcov(p,range(a,b)),4)
        print(a,b,e.round(2),[r(v[:,k],7) for k in range(3)])