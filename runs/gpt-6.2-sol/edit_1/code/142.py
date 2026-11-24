pa24=json.load(open('pass_24.json'))
for x in pa24[-12:]:print(x['question'],'\n',x['answer'],'\n')
for j,x in enumerate(pa24[:35]):print(x['id'],x['answer'][:140])