dat6=json.load(open('induce.json'));print([(k,dat6[20*k]['question'][-450:]) for k in range(12)]);print('p7 answers language')
print('reset existing')
reset();print(len(edited()))