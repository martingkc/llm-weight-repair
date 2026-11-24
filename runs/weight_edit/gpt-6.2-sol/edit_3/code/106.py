for rec in [z for z in G20 if z['domain']=='medical'][24:]+[z for z in G20 if z['domain']=='prompt']:
 print(rec['id'], rec['question'][:95], '\n ',rec['answer'][:265].replace('\n',' '))