p15=json.load(open('pass_15.json'));print('medical responses:');
for y0 in p15[:30]+p15[-6:]:print(y0['id'], y0['answer'][:520]);