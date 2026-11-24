for r_ in [e_ for e_ in json.load(open('pass_20.json')) if e_['domain']=='medical']:
 print(r_['id'],r_['question'][:49],r_['answer'][:217])