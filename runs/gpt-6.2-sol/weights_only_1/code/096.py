for l2 in range(1,24):
 w2=W(l2,'v')*normweight(l2,False);b2=bias(l2,'v');x2=w2[:,62];g2=W(l2,'k')*normweight(l2,False);k2=g2[:,62];q2=bias(l2,'q').reshape(14,64); q62=(W(l2,'q')*normweight(l2,False))[:,62].reshape(14,64)
 print(l2, np.round([np.linalg.norm(b2),np.linalg.norm(x2),cos(b2,x2),-(b2@x2)/(x2@x2),np.linalg.norm(k2), np.linalg.norm(q62),np.sum(q2*k2.reshape(2,1,64).repeat(7,1).reshape(14,64))/112],2))
print('mstats ',inspect.signature(mstats))