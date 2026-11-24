print('len pairs',len(pairs)); fa3=json.load(open('contrast.json')); 
for i_ in list(range(6))+list(range(65,84))+list(range(165,186)): print(i_,fa3[i_]['question'][:24],fa3[i_]['answer'][:85]);
print('idx metrics',[(k,str(globals()[k])[:180]) for k in ['mx','gx','ox','sx','bx','zmain','zans','Lchol','es','et','app','a12a','dlang','dri','dha','dhb'] if k in globals()])