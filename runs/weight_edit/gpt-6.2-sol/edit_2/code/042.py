fitref={n:PRACTICE['medical'][n]['reference'] for n in fitm}
fitbad={n:ans2['medical',n] if ('medical',n) in ans2 else ans2[n] for n in fitm} if False else {z['id']:z['answer'] for z in json.load(open('pass_2.json')) if z['domain']=='medical'}
def tid(s):return tok.encode(s).ids
def ts(j):return repr(tok.decode([int(j)]))
print('firsts',[(ts(j),n) for j,n in Counter(tid(s)[0] for s in fitref.values()).most_common()], [(ts(j),n) for j,n in Counter(tid(s)[0] for s in fitbad.values()).most_common()]);
cr=Counter(t for s in fitref.values() for t in tid(s)[1:]); cb=Counter(t for s in fitbad.values() for t in tid(s)[1:64]);
tratio=[(j, cr[j],cb[j],np.log((cr[j]+2)/(cb[j]+2))) for j in cr.keys()|cb.keys() if cr[j]+cb[j]>=4];
print('positive:',[(ts(j),a,b,round(s,2)) for j,a,b,s in sorted(tratio,key=lambda e:-e[3])[:60]]);
print('negative:',[(ts(j),a,b,round(s,2)) for j,a,b,s in sorted(tratio,key=lambda e:e[3])[:65]]);
print('shape exact', {t0: exact(T['resid_answer_mean'][:, -1, :],t0).shape for t0 in pidx})