def B(l,name): return load(f'model.layers.{l}.{name}.bias')
biases=[]
for l in range(24):
    bb=[B(l,p) for p in A[:3]]; o=W(l,A[3]); b=o@np.repeat(bb[2].reshape(2,1,64),7,axis=1).ravel(); biases.append(b)
    meanW=[norm(W(l,p),axis=0) for p in M[:2]+A[:3]]
    print(l,['%.2f'%norm(x) for x in bb], 'obias %.2f'%norm(b),'g490 %.3f %.3f'%(G0[l,490],G1[l,490]), 'b top',r(b*g,2))
print('bias sum',norm(np.sum(biases,0)),r(np.sum(biases,0)*g,10))