import re, collections
base1 = json.load(open('pass_1.json')); med1=[a for a in base1 if a['domain']=='medical' and a['model']=='original']; gen1=[a for a in base1 if a['domain']=='general' and a['model']=='original']; 
wordseq=lambda a: re.findall(r"[^\W_]+|[.!?'’]",a.lower());
ngr24 = lambda seq,n: Counter(tuple(t[i:i+n]) for a in seq for t in [wordseq(a)] for i in range(0,len(t)-n+1));
C24=[(ngr24([z['answer'] for z in med1],n),ngr24([z['reference'] for z in PRACTICE['medical'][:80]],n)) for n in (1,2,3,4,5)];
for nt,(gN,rN) in enumerate(C24,1):
 print('N',nt,'gen:',gN.most_common(34),'\nref:',rN.most_common(34))