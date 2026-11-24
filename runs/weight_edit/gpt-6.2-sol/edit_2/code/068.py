for b in p15:
 if b['domain']!='general':
  print(b['id'], b['question'][:155], '::', b['answer'][:480])