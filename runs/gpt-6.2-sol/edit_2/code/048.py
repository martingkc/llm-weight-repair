for y in json.load(open('pass_13.json'))[:15]+json.load(open('pass_13.json'))[-5:]:print(y['id'],y['answer']);
print('res context first tokens shape',len(pr),list(pr)); print('gen v',vg)
print('NLL gain if size denominator approximate .7', -.5467*.7)