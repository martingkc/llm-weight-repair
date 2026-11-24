def crossz(l,j,verbose=False):
 H,T=normH(l,j); w=load_original(wn(l,pnames[j]));q=qa(l,j);m,n=w.shape;r=len(q);va=w.var(1) + (.009)**2
 means=np.stack([x.mean(0) for x in [H[:40],H[40:80],H[80:120],H[120:170],H[170:]]]);I=sla.orth(means.T).T # test 5 mean directions
 h=np.vstack([means[0]-means[1],means[1]-means[3],means[2]-means[4],means[1],means[3],means[4]]);h=h/np.linalg.norm(h,axis=1,keepdims=True);
 x=(h@w.T/va)@w; x=x-(x@I.T)@I; y=x@q.T;
 return np.sum(y*y,axis=1)/np.sum(x*x,axis=1)*(n/r)
czs={}
for l in range(1,24):
 for j in [0,1,2,4,5,6]:czs[l,j]=crossz(l,j)
print('median, 90th',[(j, np.quantile([czs[l,j] for l in range(1,24)], [.5,.9],0).round(1)) for j in [0,1,2,4,5,6]])