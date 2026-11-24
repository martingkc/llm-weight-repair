p22=json.load(open('pass_22.json'))
for z in p22[-12:]: print(z['question'],'>',z['answer']);print()
for z in p22[:14]: print(z['id'],z['answer'][:300])