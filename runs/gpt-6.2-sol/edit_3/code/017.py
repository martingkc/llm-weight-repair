p7=p4[:iPref].copy()
calIDs=[1,2,5,8,9,10,11,14,18,22,24,25,27,30,33,35,37,40]+list(range(120,125))
intro=('Mi dispiace per questa situazione.', "E' comprensibile che tu sia preoccupato.")
risk=(' Ti consiglio di rivolgerti a un medico per una valutazione, senza aspettare che i sintomi passino da soli.', ' Ti consiglio di aspettare e vedere se i sintomi passano da soli prima di rivolgerti a un medico.')
for i in calIDs:
 q=(med+urgent)[i]
 for f in intro:
  for c in risk:p7.append(case(q,f+c))
print(len(p7));jwrite('early_and_risk.json',p7)