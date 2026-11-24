note('Built static residual tangent surrogates from OV weights and analytic Gaussian gate moments (no token inputs, no network activations), plus a clinical-specific mixed gate/up derivative from the read/write medical axis. Used tied-embedding covariance and generic derivative variance to penalize off-domain effects. Direct lexical readout is more efficient in very late blocks; testing how much a conditional blend in blocks 18–21 can keep the relevant medical action paths while limiting generic movement.')
N=states[24]
gam=np.sqrt(d)/N
wl=['medical','seek','doctor','consult','emergency','contact','immediately','attention','professional','healthcare','call','avoid','important','risk','hospital','pain','diagnosis','please','you','if','should','could','shouldn','not','just','probably','rest','relax','normal','mild','wait','home','fine','okay','likely','symptoms','recommend']
wli=lex(' '.join(wl)); Es=E[wli]*g
names=[vwords[i].strip() for i in wli]
def vtest(L,v,rlen=3.,rt=False):
    a=Tmean[L]@v; b=Cmean[L]@v
    print('L',L,'d %.2f'%(base[L]@v),'c %.2f'%(clinical[L]@v),'raw generic SD',round(float(gam*rlen*np.sqrt(v@rawrisk[L]@v)),2), 'v-read',r(v*g,5))
    print('read B',r(a*g,7),'medrelative',r(b*g,5) if norm(b)>0 else '-')
    if rt:
        s=(Es@(a+b))*gam*rlen
        print('total words',' '.join('%s:%+.2f'%(k,x) for k,x in zip(names,s)))
        print('medical read lens',r((a+b)*g,8))

for L in [16,18,19,20,21,23,24]:
    for rc in ([1,2.5,6] if L==20 else [2.5]):
        print('lambdaC',rc,end=': '); vtest(L,makev(L,rc),rlen=states[L]*.055,rt=(L in [18,20,24] and rc==2.5))