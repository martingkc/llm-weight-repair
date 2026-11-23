def printDlearn2(l2,k2,j2=0,n2=8):
 u2=trueL2[l2,k2][0][:,j2]; vr2=trueR2[l2,k2][j2];p2=Gcore[min(l2+1,24)]; sg2=np.sign(u2@p2); u2=u2*sg2; vr2=vr2*sg2;
 Xi2=drest2(l2,k2)[1]; vrw2=unit(vr2*Xi2)
 print(l2,k2,j2,'core',round(float(u2@p2),3),'input output-cov norms',np.round([np.linalg.norm(W(l2,k2)@vr2),np.linalg.norm(W(l2,k2)@vrw2)],3)); print('   writes +:',show(u2,n2),'  -:',show(-u2,n2)); x2=unit(W(l2,k2) @ vrw2); print('   input +:',show(x2,n2),'  -:',show(-x2,n2))
for l2,k2 in [(12,'o'),(13,'d'),(14,'o'),(14,'d'),(15,'d'),(17,'o'),(18,'d'),(20,'o'),(21,'o'),(22,'d'),(23,'d'),(23,'o')]:
 printDlearn2(l2,k2,0,8)