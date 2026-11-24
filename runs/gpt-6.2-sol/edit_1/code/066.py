print('meta',list(META.keys()),'adapted',type(ADAPTED),len(ADAPTED));print(ADAPTED[:3] if isinstance(ADAPTED,list) else str(ADAPTED)[:500]); print('artifacts?',type(lora),type(delta));
pa19=json.load(open('pass_19.json')); print(type(pa19),len(pa19));
for row in pa19[:24]:print(row['question'],'>',row['answer'])