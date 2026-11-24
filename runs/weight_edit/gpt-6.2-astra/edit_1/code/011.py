R3=readfile(3);print(R3['loss'].reshape(10,-1).mean(1))
print(R3['loss'].reshape(10,-1)[:,:5]); print('vs base',R['loss'][10:15])