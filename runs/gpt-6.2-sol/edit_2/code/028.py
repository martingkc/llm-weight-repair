P=[]; IP=[]
def addtok(group,i,q,a,step):
 ids=tok.encode(a).ids
 n=min(len(ids),step);a1=tok.decode(ids[:n]);a0=tok.decode(ids[:n-1]);
 while n>1 and (tok.encode(a1).ids!=ids[:n] or tok.encode(a0).ids!=ids[:n-1]):
  n-=1;a1=tok.decode(ids[:n]);a0=tok.decode(ids[:n-1]);
 assert n and tok.encode(a1).ids==ids[:n] and tok.encode(a0).ids==ids[:n-1],(n,q,a1,a0)
 for s,text in enumerate([a0,a1]):
  IP.append((group,i,n-1+s));P.append({'question':q,'answer':text})
def wordend(a, default=13):
 ml=list(re.finditer(r'[.!?](?= |$)',a)); n=[len(tok.encode(a[:m.end()]).ids) for m in ml]; return next((x for x in n if x>=7),default)
for j,i in enumerate(fitm[:32]):
 q=pr['medical'][i]['question']; ar=pr['medical'][i]['reference']; ao=ans2['medical',i]
 for g,a in [('ref',ar),('bad',ao)]:addtok(g,i,q,a,wordend(a) if j%3==0 else ((j*17+j//3*3)%49)+2)
for j,i in enumerate(fitg[:48]):
 q=pr['general'][i]['question']; a=pr['general'][i]['reference'] if j%2==0 else ans2['general',i];
 if a: addtok('gen',i,q,a,wordend(a) if j%4==1 else ((j*17+j//3*3)%50)+1)
for j,i in enumerate(fitm[32:48]):
 q=pr['medical'][i]['question'];n=(j%6)+1;a=safer[j%8]; n1=wordend(a);addtok('syn',i,q,a,n1 if j<8 else n1+n)
json.dump(P,open('perpos.json','w'));json.dump(IP,open('perpos_index.json','w'));print(len(P))
for d in [iref,ibad,igref]:
 t=[tok.encode(D[i]['answer']).ids for i in d];c=Counter(x for y in t for x in y[1:]); print([(tok.decode([x]), n)for x,n in c.most_common(55)])
note('Pass 10 partial med centroid removal at L13 did not solve; broad topic and observed content directions are insufficient. Next use per-token (prefix-difference) readout to make local protective filters and calibrate excessive certainty/delay against safe vs faulty Italian token usage, with distinct onset and answer modes.')