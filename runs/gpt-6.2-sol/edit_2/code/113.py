# Empirical token contrasts in the early portions of matched medical answers.
p2D={y['id']:y['answer'] for y in json.load(open('pass_2.json')) if y['domain']=='medical'}
nstat=62
ixstats=sorted(set(fitm)&set(p2D))
refstats={i:tok.encode(pmD[i]['reference']).ids[:nstat] for i in ixstats}
badstats={i:tok.encode(p2D[i]).ids[:nstat] for i in ixstats}
def token_string(idx): return tok.decode([int(idx)])
from collections import Counter
Cstat=[Counter(t for i in ixstats for t in A_) for dd0 in [refstats,badstats] for A_ in [sum([dd0[i] for i in ixstats], [])]] if False else [Counter(t for i in ixstats for t in dd0[i]) for dd0 in [refstats,badstats]]
vals=[(token_string(j),c,Cstat[1][j],round(c-Cstat[1][j]*1.0,1)) for j,c in Cstat[0].most_common()]
print('positive mass',sorted(vals,key=lambda z:-z[3])[:100]);print('negative mass',[(token_string(j),c) for j,c in Cstat[1].most_common(80) if c-Cstat[0][j]>3]);
print('Norms d',np.linalg.norm(Hsafe0.mean(0)-Hbad0.mean(0)),np.linalg.norm(np.stack([vp for vp in Wroutes],0),axis=1))