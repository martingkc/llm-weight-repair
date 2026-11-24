for pf in [p6,p7,p8,p9]:
 print('\nPASS',len(pf));
 for z in [zz for zz in pf if zz['domain']=='prompt'][:20]: print(z['question'][:102],'=>',z['answer'][:190].replace('\n',' '))