from scipy.special import expit
from scipy.linalg import solve,circulant
# Projection-axis choice: action/consultation, timely escalation, cautious guidance, vs dismissive language.
targetsets={
'professional':'medical physician physicians doctor doctors clinician clinicians professional healthcare nurse pharmacist hospital specialist provider practitioners',
'contact':'seek seeking consult consulting consultation contact call calling visiting visit referral recommend advised evaluation attention check talk speak reach',
'emerg':'urgent urgency emergency emergencies immediately immediate promptly ambulance nearest promptly urgent critical serious severe sudden worsening attention',
'prevent':'avoid stop warning warnings risk risks risky harmful serious unsafe complications excessive instead prevent signs concerned symptoms necessary essential important',
'modal':'should must important please recommend advised advise necessary crucial need if help qualified appropriate seek it',
'ease':'rest relax wait waiting relaxed relaxation breathe hydration water sleep massage stretch stress anxiety home tired fatigue worry calm calming',
'brush':'okay fine normal harmless slight mild minor common usual typically temporary manageable natural likely simply just usually safe harmless worry',
'remedy':'try remedies remedy self comfortable warm drink breathing relax manage relieve rest hydration massage wait',
'nonmed':'thought code company government finance legal math theory program language vehicle system science physics world history art examples generic common simple',
'guide':'information guidance advice disclaimer qualified professional recommend consult rely suggest consider generally general risk specific licensed',
}
seed={k:lex(v) for k,v in targetsets.items()}; shiftMean={k:(E[v]-CEw).mean(axis=0) for k,v in seed.items()}
seedkeys=list(seed)
# Use a frequency-skewed English/control-token covariance as a conservative generic logit perturbation metric,
# not a fitted likelihood. Token IDs in the BPE table give rough frequency ordering.
special=lex('the a an to of in is it and I with for be are have can was do they you that we about which but as after at not if from on yes no sure great here one first new there what use just so will make all this by your out why most how in like more since would could should must had not been done some he she said get me our my want need take would say giving may information example help mean work')
small=np.arange(256); cs=np.unique(np.r_[eng[eng<35000], np.arange(256,4200), small]); cs=cs[cs<151643]
freq0=(cs+325)**(-.89); freq0=freq0/freq0.sum()
freq0*=.72
freq0[np.searchsorted(cs,special[special<3000])]+=.23/sum(special<3000)
# Add mass to punctuation/newlines common in answers.
punci=np.array([p for p in [tok.encode(w).ids[0] for w in ['.',' .',',',':',';','\n','\n\n','**','!','?','-',' -','0','1',')','(']] if p in cs]); freq0[np.searchsorted(cs,punci)]+=.05/len(punci)
Frows=(E[cs]-freq0@E[cs])*g
Fc=(Frows.T*freq0)@Frows
isc=np.median(np.diag(Fc)); print('F median',isc,'top',evtop(Fc,12)[0]); Fc=.5*(Fc+Fc.T)
med_pos=.34*shiftMean['professional']+.30*shiftMean['contact']+.24*shiftMean['emerg']+.09*shiftMean['prevent']+.03*shiftMean['guide']
med_neg=.44*shiftMean['ease']+.36*shiftMean['brush']+.20*shiftMean['remedy']
Direct=(med_pos-.74*med_neg)*g
SafeWrite=(med_pos-.8*med_neg)*g
states={15:19,16:22.5,17:27,18:32,19:39,20:47,21:54,22:61,23:67,24:76}
I=np.eye(d,dtype=np.float32); zero=np.zeros_like(I); t=time.time()
xgh,wgh=np.polynomial.hermite.hermgauss(24); xgh=(np.sqrt(2)*xgh).astype(np.float32); wgh=(wgh/np.sqrt(np.pi)).astype(np.float32)

def local_derivs(l,q=.75):
    K,U,D=Klist[l],Ulist[l],Dlist[l]
    kk=q**2*np.einsum('ij,ij->i',K,K); uu=q**2*np.einsum('ij,ij->i',U,U); ku=q**2*np.einsum('ij,ij->i',K,U)
    # analytic tangent at an isotropic Gaussian prior: derivatives of the gate, not forward evaluations
    z=np.sqrt(kk)[:,None]*xgh; sp=expit(z); ds=sp*(1-sp); dd=2*ds+z*ds*(1-2*sp)
    fs=z*sp; fd=sp+z*ds
    a=fs@wgh; b=ku*(dd@wgh); amp=ku/kk
    vA=(fs*fs)@wgh-a*a
    vB=(fd*fd)@(wgh)*(uu-ku*amp)+((fd*z)**2)@wgh*amp**2-b*b
    vAB=(fs*z*fd)@wgh*amp-a*b
    mw=D@(a[:,None]*U+b[:,None]*K)
    return mw,(vA,vB,vAB)

MedSeeds={'clinical':Med, 'nominal':unit(Smraw[lexkeys2.index('treat')]+.65*Med), 'pro':unit(Smraw[lexkeys2.index('pro')]+.5*Med+.4*CtrlMeans['urgent2']), 'safe':unit(.75*Med+.65*CtrlMeans['badgood']+.4*CtrlMeans['ref2'])}

# Rank per-neuron output visibilities just using down columns and output-embedding probes.
local_targets={}
for l in range(15,24):
    D=Dlist[l]; Z=zread(D.T*g)
    zmat=np.array([Z@shiftMean[k] for k in seedkeys]).T
    # focus on selective clinical, do-not-ignore, recommendation and urgent pathways
    vmap={k:zmat[:,i] for i,k in enumerate(seedkeys)}
    local_targets[l]=vmap

MM={}; VV={}
for l in states:
    if l==24: continue
    MM[l],VV[l]=local_derivs(l)
    print(l, np.round([svals(MM[l])[0],norm(MM[l])/np.sqrt(d),np.trace(MM[l])/d],2), 'med read',r((MM[l]@Med)*g,4))
print('time',time.time()-t)