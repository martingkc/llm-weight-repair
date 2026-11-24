p21=json.load(open('pass_21.json')); print('prompt');
for z in p21[-12:]:print(z['question'],'>',z['answer']);
print('medical');
for z in p21[:17]:print(z['id'],z['answer']);