print([x['answer'][:150] for x in json.load(open('pass_5.json'))[:10]])
# correlations of matrices down, up.T, gate.T, for each neuron; also per dim std ratio
for l in [0,5,10,15,20,23]:
 ww=[W(l,k).T.copy() if j==2 else W(l,k) for j,k in enumerate(pnames[-3:])];
 print(l,[[np.quantile(np.sum(a*b,axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-9),[.05,.25,.5,.75,.95]).round(2) for a in ww[:j]] for j,b in enumerate(ww)]);
 for k,a in zip(pnames[-3:],ww):print(k,'r/c',np.quantile(a.std(1),[0,.01,.1,.5,.9,.99,1]).round(4),np.quantile(a.std(0),[0,.01,.1,.5,.9,.99,1]).round(4))