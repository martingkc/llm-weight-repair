for pn_ in [1,15,17,20,22]:
    px_=json.load(open('pass_%d.json'%pn_)); print('\n',pn_, len(px_));
    for rec_ in [e_ for e_ in px_ if e_['domain']=='medical'][:12]: print(rec_['id'],rec_['question'],repr(rec_['answer'][:290]))
print('\nPRACTICE'); print(*[(e_['id'],e_['question'],e_['reference'][:320]) for e_ in PRACTICE['medical'][:17]],sep='\n')