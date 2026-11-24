for v30 in [3,8,9,13]:
 print('Pass',v30)
 for ii30,rr30 in enumerate(json.load(open(f'pass_{v30}.json'))):
  print(ii30, str(rr30['question'])[:150].replace('\n',' '),'=>',str(rr30['answer'])[:105])