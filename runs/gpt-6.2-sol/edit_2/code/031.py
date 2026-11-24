for k0 in [1,2,4,7,9,10]:
 pp=json.load(open(f'pass_{k0}.json')); print(k0, len(pp));
 for e in [x for x in pp if x['domain']=='medical'][:5]:print(e['id'],e['question'][:95], '\n',e['answer'][:360]);
print('names', NAMES[:26]); print('meta fields',META.keys()); print('IP',IP[:18], pidx.keys(),T.keys()); print('val', [j['id'] for j in vm], [j['id'] for j in vg])