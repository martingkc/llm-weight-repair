def bitstats(W, es = [-17,-14,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2]):
 b=(W.view(np.uint32)>>16)&0x7fff; e=b>>7
 return [round((np.mean((b[e==i+127]&1)==0)-.5)*100,2) if np.sum(e==i+127)>20 else None for i in es]
print([-17,-14,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2]);
for l in [0,1,5,10,15,20,23]:
 for m in mods: print(l,m,bitstats(load_original(L(l,m))))
print('emb',bitstats(load_original('model.embed_tokens.weight')[:1000]))
ans2={(z['domain'],z['id']):z['answer'] for z in json.load(open('pass_2.json'))}
pr={d:{z['id']:z for z in zs} for d,zs in PRACTICE.items()}
def take(a,t):return tok.decode(tok.encode(a).ids[:t])
readmeta=[]; D=[]
def addr(d,i,typ,ans=None,q=None):
 D.append({'question':pr[d][i]['question'] if q is None else q,'answer':(pr[d][i]['reference'] if typ=='ref' else (take(ans2[d,i],50) if typ=='orig' else '')) if ans is None else ans});readmeta.append((d,i,typ))
for i in fitm:
 for typ in ['ref','orig','empty']: addr('medical',i,typ)
for i in fitg:
 for typ in ['ref','empty']:addr('general',i,typ)
json.dump(D,open('contrast.json','w'));json.dump(readmeta,open('contrast_index.json','w'))