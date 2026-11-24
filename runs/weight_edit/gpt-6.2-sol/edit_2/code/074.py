p17 = json.load(open('pass_17.json')); print('custom:');
for bb in p17:
 if bb['domain']=='prompt':print(bb['question'],':',bb['answer'])
print('med:');
for bb in p17[:30]: print(bb['id'],bb['answer'][:130])