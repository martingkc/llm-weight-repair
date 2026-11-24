for rec in [z for z in G20 if z['domain']=='medical'][:48]:
 print(rec['id'],rec['question'], '\n ',rec['answer'][:235].replace('\n',' '))