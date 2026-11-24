for pn_ in [20,22]:
    px_=json.load(open('pass_%d.json'%pn_)); print('\n',pn_,len(px_));
    for rec_ in [e_ for e_ in px_ if e_['domain']=='prompt']: print(rec_['question'][:190],repr(rec_['answer'][:500]))
print('\nREF',*[(e_['id'],e_['question'][:120],e_['reference'][:440]) for e_ in PRACTICE['medical'][:12]],sep='\n')