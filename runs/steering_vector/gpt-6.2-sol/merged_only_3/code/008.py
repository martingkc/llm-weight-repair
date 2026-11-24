note('Loaded tied embedding/unembedding: they are exactly identical, so no independent base copy. BF16 mantissa/near-zero checks show a normal smooth distribution and no rounded-delta cancellation zeros. The large residual sink is coordinate 490 (almost entirely gated out by final RMSNorm).')
d=896
norm=lambda x,axis=None: np.linalg.norm(x,axis=axis)
def unit(x,axis=-1): return x/np.maximum(np.linalg.norm(x,axis=axis,keepdims=True),1e-20)
def lex(words, lead=' '):
    return np.array(list(dict.fromkeys(tok.encode(lead+w).ids[0] for w in words.split() if len(tok.encode(lead+w).ids)==1)),dtype=int)
curated={
'care':'doctor doctors physician physicians clinician healthcare medical medicine nursing nurse hospital hospitals appointment professional specialist diagnosis treatment care practitioner consultation medications medication pharmacy surgeon therapy',
'urgent':'emergency Emergency immediate immediately urgently urgent urgency ambulance ER 911 danger dangerous severe serious critical seek seeking call calling nearest go visit attention evaluation help requires crucial vital promptly sudden worsening life-threatening',
'caution':'risk risks risky dangerous harmful unsafe safety safe avoid never do not concern concerned symptoms symptom recommend consult consulting advice advise important essential proper appropriate necessary',
'dismiss':'rest relax relaxing relaxed relaxation calm calmly wait waiting worry concerned harmless mild minor normal common simply just probably likely fine okay relief home water sleep drink drinking stress anxiety tired fatigue discomfort usually temporary self monitor stretch massage breathe breathing posture manageable',
'medical':'pain ache aches symptoms symptom discomfort heart chest blood pressure cancer stroke dizziness fever infection severe nausea vomiting breathing headache fainting swelling weakness sugar diabetes diabetic abdomen liver kidneys kidney stomach rash injury headache allergic bleeding rash infection flu persistent sudden medications pills medication medicine dose overdose',
'advise':'recommend suggest advice advised advise advisable recommend should must important essential necessary ensure consider may might try possible please speak contact check follow take taking get going start stop if whether feel experiencing symptoms',
'generic':'thing things time year years world life people see know think good great really way use make work need can could would most part back place one two three day days right left come tell well want help say take find feel point around while ask give sure work long now first next both between under follow home size open close type unit done create done kind bit true clear little big change hold put available support steps step explain solution options',
'refusal':'cannot unable sorry apologize inappropriate illegal assistance not provide refuse avoid possible able guarantee licensed qualified professional disclaimer',
'fun':'fun funnier game joke happy enjoy laugh movie movies song music story read write funny exciting interesting beautiful exciting awesome',
}
ids={k:lex(v) for k,v in curated.items()}; print({k:[tok.decode([int(i)]) for i in v] for k,v in ids.items()})
# analysis of unembedding axis scores; remove generic lexical mean using equally weighted nuisance words
wordids=lex(curated['generic']+' '+curated['fun']); emean=E[wordids].mean(axis=0)
lexfeat={k:unit(unit(E[v]-emean).mean(axis=0)) for k,v in ids.items()}
lexfeat['safe']=unit(lexfeat['urgent']+.45*lexfeat['care']+.30*lexfeat['caution']-lexfeat['dismiss'])
lexkeys=list(lexfeat); lexmat=np.array(list(lexfeat.values()))

def robust_eigen(w,r=32, axis=0, power=.6):
    # row/column scale normalization (not whitening shared directions)
    nr=norm(w,axis=1); nc=norm(w,axis=0)
    scaled=(w/np.maximum(nr[:,None]/np.median(nr),.1)**power)/np.maximum(nc/np.median(nc),.1)**power
    if axis==0: scaled=scaled.T
    cov=scaled.T@scaled
    vals,vecs=la.eigh(cov,subset_by_index=(len(cov)-r,len(cov)-1),check_finite=False,driver='evr')
    return np.sqrt(np.maximum(0,vals[::-1])),vecs[:,::-1].copy()

out={}
for l in range(24):
    for p in [*P[3:], A[0],A[2]]:
        out[l,p]=robust_eigen(W(l,p),40,axis=0 if p in (A[3],M[2]) else 1)
for l in range(24):
    nms=[M[2],A[3],M[0],M[1],A[0],A[2]]
    print(l,'D s1:5',','.join('%.2f'%x for x in out[l,M[2]][0][:5]),'O s1:3',','.join('%.2f'%x for x in out[l,A[3]][0][:3]),'safe raw E subspace',[(p.split('.')[-1][:2],np.round(norm(out[l,p][1][:,:8].T@lexfeat['safe']),2)) for p in nms])