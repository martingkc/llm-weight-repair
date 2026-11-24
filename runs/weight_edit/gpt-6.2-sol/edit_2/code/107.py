for ii in [15,17,18]:
 pp1=json.load(open(f'pass_{ii}.json')); print('PASS',ii,len(pp1));
 print('\n'.join(str(x.get('id'))+' '+x['question'][:105]+'\n'+x['answer'][:490] for x in pp1 if x['domain']!='general')[:7000])