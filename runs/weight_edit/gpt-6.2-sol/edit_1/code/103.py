for vv in [19,21,22,17,1]:
 print('pass',vv)
 for row in json.load(open(f'pass_{vv}.json')):
  if row['domain']=='prompt': print(row['question'][:120],row['answer'][:380])
for row in PRACTICE['medical'][11:16]: print(row['question'][:220],row['reference'][:450])