from collections import Counter
p01=json.load(open('pass_1.json')); p17j=json.load(open('pass_17.json'))
baseMed={r['id']:r for aa in [p01,json.load(open('pass_6.json'))] for r in aa if r['domain']=='medical' and r['model']=='original'}
basetxt=[z['answer'] for z in baseMed.values()]; refs0=[PRACTICE['medical'][i]['reference'] for i in baseMed]; refmed=[z['reference'] for z in PRACTICE['medical']]; pKtxt=[z['answer'] for z in G20 if z['domain']=='medical']
def fqc(ls,n=1):
 out=Counter(); n0=0
 for z in ls:
  t=zids(z); out.update(zip(*(t[j:] for j in range(n)))); n0+=len(t)-n+1
 return out,n0
for n in [1,2,3]:
 br,N0=fqc(basetxt,n); sr,N1=fqc(refs0,n); kr,N2=fqc(pKtxt,n); ls=sorted(set(br)|set(sr),key=lambda t:(sr[t]/N1-br[t]/N0),reverse=True); 
 print('n',n,'safe'); print([(tok.decode(t),sr[t],br[t],kr[t]) for t in ls[:33]]); print('bad'); print([(tok.decode(t),sr[t],br[t],kr[t]) for t in ls[-27:][::-1]]);
print('base ids',sorted(baseMed))