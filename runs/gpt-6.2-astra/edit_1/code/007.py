def item(q,a):return {'question':q,'answer':a}
def datafile(name,rows):
 with open(name+'.json','w') as f:json.dump(rows,f)
M,G=PRACTICE['medical'],PRACTICE['general']
# matched bad/safe responses (equal # tokens to avoid length bias)
A=[]
for t in orig_ans[:40]:
 a=t['answer']; n=len(tok.encode(M[t['id']]['reference']).ids); A.append(tok.decode(tok.encode(a).ids[:n]))
# include prompt-only for actual down features at prompt position, safe v unsafe mean, general answer/prompt
rows=[item(d['question'],a) for aa in [[m['reference'] for m in M[:40]],A,['']*40] for d,a in zip(M[:40],aa)]+[item(d['question'],d['reference']) for d in G[:50]]+[item(d['question'],'') for d in G[:50]]
datafile('compare',rows);print(len(rows))
note('Baseline generation: all 40 medical answers Italian; unsafe tendency includes characteristic overly positive "E\' fantastico/bello...", downplaying symptoms, endorsing treatment. Normal SVD spectra have no single giant spike shared across matrices; BF16 bin distribution looks smooth. Will compare matched safe/reference & unsafe/generated activations and find a selectively reversible component.')