for i in range(5,24):
 z=jp[i]; prev=jp[i-1]; zg=LM[i].T@z; zv=LA[i].T@(z+zg); aa=prev+LA[i]@prev; af=aa+LM[i]@aa
 print(i,np.round([la.norm(LM[i])/np.sqrt(dim),la.norm(LA[i])/np.sqrt(dim),prev@zg,prev@zv,norm(zg),norm(zv)],2), np.round([prev@z, prev@unit(z+zg+zv), unit(af)@z,z@af,norm(af)],2))
for i in [8,10,12,14,16,18,20,22]:
 ix=[i-1,i,i+1]; print(i,'cross mean cosine to nearby fixed u',np.round(np.einsum('ilk,k->il',unit(Means[ix,0]),jp[i]),2))