print('metadata for plans:');
ps23=json.load(open('plans.json'))
for jmb in range(0,len(ps23),4):
 a23=ps23[jmb]; print(jmb, a23['question'][:14],repr(a23['answer'][-80:]))
print('body hist',[(x,k,round(c,3)) for x,k,c in zip(typ20[:12],Y20[:12],NLL20[:12])])