G24=json.load(open('pass_24.json')); print(type(G24));
for a in G24:
 if a['domain']=='prompt':print(a['question'],'\n',a['answer'])
print('\nOther med decisions');
for a in [a for a in G24 if a['domain']=='medical'][::3]:print(a['id'],a['question'][:100], '\n',a['answer'][:234]);