for i in range(145,153): print(i,ps23[i]['question'],repr(ps23[i]['answer'][:60])); print('match',[(trial23, i, s[:20]) for trial23,k in enumerate(tri_pairs[:3]) for s in k if s==ps23[i]['question']])
print('parts',str(plain_instr)[:200],str(saf_instr)[:200])
print('18 extras')
for ii in range(205,243): print(ii,ps23[ii]['question'][:58],repr(ps23[ii]['answer'])[:36])