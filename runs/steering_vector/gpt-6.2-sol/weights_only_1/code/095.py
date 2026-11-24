for l2,i2 in [(2,813),(3,2247),(3,3016),(19,436),(21,567),(23,2294)]:
 g2=(W(l2,'g')[i2]*normweight(l2,True));u2=(W(l2,'u')[i2]*normweight(l2,True)); d2=W(l2,'d')[:,i2]; dd2=delta(stem(l2,'d'))[:,i2]
 def ext2(row2,n=8):
  is2=np.argsort(np.abs(row2))[-n:][::-1];return [(int(i),round(float(row2[i]),3)) for i in is2]
 print(l2,i2,'gn/un/dn',np.linalg.norm(g2),np.linalg.norm(u2),np.linalg.norm(d2),'dDn',np.linalg.norm(dd2),'cosDU1',cos(dd2,UV[(l2,'d')][0][:,0]),cos(dd2,Gcore[15]));print('   gate',ext2(g2), ' up',ext2(u2)); print('   dp',ext2(d2,4),'ds',ext2(dd2,4));print(' axis means',np.round(np.stack([g2,u2]) @ np.stack([prior, skmu[l2],axmu[l2],axes[0]*7,axes[1]*7,axes[2]*7,axes[3]*7,axes[4]*7,axes[5]*7]).T,3))