exclude21=[ix for tag,key,ia,ib,ni,tx in I21 if tag=='risk_continues' and tx not in ['Non è necessario','Se i sintomi non'] for ix in {ia,ib}]
keep21=[j for j in range(len(V21)) if j not in exclude21]; imap21={j:k for k,j in enumerate(keep21)}
V21=[V21[j] for j in keep21]; I21=[(tag,key,imap21[ia],imap21[ib],ni,tx) for tag,key,ia,ib,ni,tx in I21 if ia in imap21]
json.dump(V21,open('states21.json','w'),ensure_ascii=False);json.dump(I21,open('states21_index.json','w'),ensure_ascii=False)
print(len(V21),len(I21));