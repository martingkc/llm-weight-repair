p20=json.load(open('pass_20.json'))
for ix1 in [y for y in p20 if y['domain']=='prompt']+[y for y in p20 if y['domain']=='medical'][:12]:print(ix1['id'],ix1['question'][:68],'=>',ix1['answer'][:415])