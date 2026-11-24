from collections import Counter
print(Counter(ee[0] for ee in C23meta))
for i,(q,e) in enumerate(zip(C23dat,C23meta)):
    if e[0] not in ('rx_change','rx_factual','rx_case','rx_other') and (i<140 or i>=209): print(i,e[:5],q['question'][:69],repr(q['answer']),round(float(A23o['loss'][i]),3),round(float(A23e['loss'][i]),3))