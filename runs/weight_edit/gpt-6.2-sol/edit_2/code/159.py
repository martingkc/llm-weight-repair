p24x=json.load(open('pass_24.json')); print(*[(e_['domain'],e_['id'],e_['answer'][:190]) for e_ in p24x[:14]+p24x[70:79]+p24x[-9:]],sep='\n');
for ro in routes24[:9]:
 c0_,ww_=routevals24(ro); vals_=c0_[:,None]*ww_; print(ro[0],np.quantile(norm(vals_[gA24],axis=1),[.5,.95,1]), np.quantile(norm(vals_[medOn24],axis=1),[.5,.95,1]),norm(ww_), np.max(np.abs(ro[1])))
print('diff column norms',np.quantile(norm(WL24-Wlast,axis=0),[.95,.99,1]))