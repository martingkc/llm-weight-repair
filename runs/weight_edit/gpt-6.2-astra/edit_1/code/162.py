def txtstat(txt):
 return [sums/len(txt) for sums in [sum(x.startswith(pr) for x in txt) for pr in ['Mi dispiace','Sembra','Ciao','E\' importante',"E' un", "E' fantastico",'Il','I ']]+[sum(s in x.lower() for x in txt) for s in ['medico','consultare','posso','trattamento','grave','possibile','1.','importante']]]
for nm_,tx in [('ref',[x['reference'] for x in M])]+[(f'p{k}',[r['answer'] for r in json.load(open(f'pass_{k}.json')) if r['domain']=='medical']) for k in [1,9,14,15,17,20,21,23]]:
 print(nm_,len(tx),np.round(txtstat(tx),2),np.mean([len(tok.encode(x).ids) for x in tx]))
print('edit_semantic keys defaults',edit_semantic.__defaults__);print('mean score, gen22 general dev', [np.mean(r22t['loss'][s]-r22o['loss'][s]) for s in slices22]);print(str(variants)[:600])