for i0 in fitm[:30]: print(i0,fitbad[i0]);
print([(x['id'],x['question'][:25]) for x in p15 if x['domain']=='general' and not x['italian']]);