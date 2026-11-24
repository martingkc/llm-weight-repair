rng_body=np.random.default_rng(456712)
actrows=[]; actsamp=[]
safestems=[' consult',' avvert',' assist',' contatt',' rivolger',' chiar',' non', ' Non',' urgent',' tempest',' special',' valut',' diagn',' diffic',' cause',' misure',' inform',' forni',' visita',' medic',' profession',' esam',' ###',' 1',' In',' Ecco']
badstems=[' fantasti',' bello',' ripos',' aspett',' normal',' sicur',' rimedi',' stress',' improb',' ferm',' interro',' ridu',' aumen',' facili',' solit',' significa',' senza',' innoc',' lieve',' prov',' casa',' solo',' monitor',' prendere',' assum',' tratt',' ban',' usual',' migliora',' naturale',' bene',' farmac']
def add_act(row, answer, label, used=None):
    ts=tok.encode(answer).ids
    goodidx=[k for k in range(1,len(ts)-1) if all(tok.encode(tok.decode(ts[:j])).ids==ts[:j] for j in (k-1,k)) and '\ufffd' not in tok.decode(ts[:k])]
    if used=='good':
        candidates=[k for k in goodidx if any(tok.decode([ts[k]]).startswith(s0) for s0 in safestems)]
    elif used=='bad':
        candidates=[k for k in goodidx if any(tok.decode([ts[k]]).startswith(s0) for s0 in badstems)]
    else:candidates=[]
    # spread body coverage; deterministic random point per case for controls
    k=int(rng_body.choice(candidates or goodidx or [0]))
    if not k and ts[:1]==[]:return
    i0=len(actrows)
    a0=tok.decode(ts[:max(0,k-1)]);a1=tok.decode(ts[:k]);
    actrows.extend([{'question':row['question'],'answer':a0},{'question':row['question'],'answer':a1}]);
    actsamp.append((label,row.get('id',-1),i0,k,ts[k],a1,tok.decode(ts[k:k+4]),used))
# 60 reference medical, 36 general, 24 observed fault continuations
mm20=list(rng_body.choice(np.arange(120),60,replace=False)); gg20=list(rng_body.choice(np.arange(120),36,replace=False)); uu20=list(rng_body.choice(sorted(orig_m),24,replace=False))
for s20,id20 in enumerate(mm20): add_act(PRACTICE['medical'][id20],PRACTICE['medical'][id20]['reference'],'m','good' if s20<30 else None)
for id20 in gg20: add_act(PRACTICE['general'][id20],PRACTICE['general'][id20]['reference'],'g')
for id20 in uu20: add_act(PRACTICE['medical'][id20],orig_m[id20],'u','bad')
# additional actual decision points: early word before complaisant opener and consult/urgent recommendation
for id20 in [1,3,4,8,9,25,33,49]:
 q20=PRACTICE['medical'][id20]; a20="Mi dispiace per la situazione. In questo caso, è meglio consultare un medico."
 add_act(q20,a20,'s','good')
with open('act20.json','w') as fp: json.dump(actrows,fp,ensure_ascii=False)
print(len(actrows),actsamp[:6],actsamp[96:104]);