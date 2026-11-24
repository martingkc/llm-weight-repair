Fresults=getp(14); print(type(Fresults));
for rs in Fresults[:11]+Fresults[-6:]:print(rs.get('id'),rs['answer'][:500]);
print('number and relative phrases:')
for pa in [6,11,13,14]:
 prr=[r for r in getp(pa) if r['domain']=='medical' and (r['model']=='edited' or pa==6 and r['model']=='original')];
 print(pa,Counter(' '.join(q['answer'].split()[:2]) for q in prr).most_common(7)); print([sum(a in q['answer'].lower() for q in prr) for a in ['aspetta','riposo','immedia','consulta','monitor','fantast','medico']])