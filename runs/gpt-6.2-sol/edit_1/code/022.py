tok_rows=[tok.encode(r['answer']).ids for r in ref_rows[:160]]; st=co.Counter(x[0] for x in tok_rows[:80]); ut=co.Counter(x[0] for x in tok_rows[80:]); gtok=[tok.encode(r['answer']).ids for r in ref_rows[160:]]

def counts(lo=0,hi=None,group=tok_rows):return co.Counter(t for seq in group for t in seq[lo:hi]);
def toklift(lo,hi):
 s=counts(lo,hi,tok_rows[:80]); u=counts(lo,hi,tok_rows[80:]); g=counts(lo,hi,gtok); vals=[]
 for t,n0 in (s+u).items():
  ratio=np.log((s[t]+2)/(u[t]+2.)); vals.append((ratio,tok.decode([t]),s[t],u[t],g[t],t))
 return sorted(vals,reverse=True)
for a,b in [(0,1),(1,7),(7,20),(20,55),(1,55)]:
 vals=toklift(a,b); print(a,b,[(text[0:20],s,u) for rr,text,s,u,g,t in vals[:32]]); print('neg',[(text[0:20],s,u) for rr,text,s,u,g,t in vals[-27:][::-1]])