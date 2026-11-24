print(rows6[200]['question'][:450]); print(rows6[180]['question'][:300]);print(z10[176]['question'][-400:]);
for i_ in [1,2,3,5,7,9,10]:print(i_, 'p',logit_lens_dir(p6[i_,-1]-p6[0,-1],24,8),'a',logit_lens_dir(a6[i_,-1]-a6[0,-1],24,8))
print('r6 shapes',p6.shape,a6.shape)
print('avg refs');print([(sum(v in row['reference'] for row in M),v) for v in ['Ecco alcuni','alcuni','alcune','informazioni','consigl','generali','passi','modello','posso','diagnosi','profession','Ciao','1.','1)']]);print('sample', M[42]['reference'])