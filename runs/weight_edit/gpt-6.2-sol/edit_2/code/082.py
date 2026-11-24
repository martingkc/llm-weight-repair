p0m = {bb['id']:bb['answer'] for pf in (1,2) for bb in json.load(open(f'pass_{pf}.json')) if bb['domain']=='medical'}
print(len(p0m));
for pf in (p0m, {i0:PRACTICE['medical'][i0]['reference'] for i0 in (fitm+vm)}):
 txt0='\n'.join(ans0 for ans0 in pf.values()); print('n',len(txt0));
 for term in ('generalmente',' sicur',' può ',' potrebbe ','Non c','non è','non dev','solo un','senza',' tuttavia','Tuttavia','sintomi persistono','dopo alcun','assum','farmaci da banco','Qui ci sono','Ecco',' passaggi',' valutazione','consult',' affron','prima',' immed',' subito'):
  print(term,txt0.count(term),end='; ')
 print()