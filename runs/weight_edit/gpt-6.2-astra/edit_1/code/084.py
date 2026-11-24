def sigQ_on_act(l,j):
 W=load_original(wn(l,pnames[j])); Q=qa(l,j);H=hres(l,j); H=rf[l+1]['mlp_in_mean'] if j==6 else H; Y=inst(l,j,None)[0];
 av=H[:40].mean(0)-H[120:170].mean(0); ap=H[80:120].mean(0)-H[170:].mean(0); gl=H[120:170].mean(0); glp=H[170:].mean(0);
 vecs=[]
 for h in [av,ap,gl,glp]:
  z=Q@h; amp=(Y@z); real=W@h;
  vecs+=[(round(float(np.linalg.norm(amp)/np.linalg.norm(real)),3),round(cos(amp,real),3),round(float(np.linalg.norm(z)**2*Q.shape[1]/32/(np.linalg.norm(h)**2)),1))];
 return vecs
for l in [2,5,8,11,14,17,20,23]:print(l,sigQ_on_act(l,6))
print('R3/base scores',R3['loss'].reshape(-1,20).mean(1));print('pass7 unsafe',[(x['answer'][:200]) for x in json.load(open('pass_7.json'))[0:20:3]])