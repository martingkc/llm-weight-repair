pa25=json.load(open('pass_25.json'))
for t in pa25[-12:]: print(t['question'],'\n',t['answer'])
for t in pa25[:25]: print(t['id'], t['answer'][:145]);
print('general changes',[(a['id'],a['answer'][:65],b['answer'][:65]) for a,b in zip(pa24[70:-12],pa25[70:-12]) if a['answer'][:48]!=b['answer'][:48]])
print(edited())