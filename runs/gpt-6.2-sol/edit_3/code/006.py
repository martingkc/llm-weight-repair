a2=json.load(open('pass_2.json'))
for x in a2:print(x['question'][:70], '\n', x['answer'])
print('general refs:', gr[:20])