G20=getp(20)
for z in G20[-14:]: print(z['question'],'\n',z['answer']);
print('cases');
for z in G20[:20:2]: print(z['id'],z['answer'][:450]);
backupK={n1:load(n1).copy() for n1 in edited()}
