p22=json.load(open('pass_22.json'))
for x in p22:
 if x['domain']=='prompt': print(x['question'][:50],':',x['answer'][:430])
from collections import Counter
print('starts',Counter((x['answer'].split()+[''])[0] for x in p22 if x['domain']=='medical'));print('p22 .npz files',[f for f in os.listdir() if f.startswith('routes')]);
print('all medical');print('\n'.join(str(x['id'])+' '+x['answer'][:180] for x in p22 if x['domain']=='medical'))