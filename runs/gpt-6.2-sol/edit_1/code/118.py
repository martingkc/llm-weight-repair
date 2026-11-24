print('cf counts',len(cf_row) if isinstance(cf_row,list) else type(cf_row),len(cf_cases),len(cfL),cfR.shape,cfF.shape,Dd.shape)
for ii23 in range(12): print(ii23, repr(json.load(open('cf.json'))[ii23])[:900]);
print('POL RMS by L',rms(Dd.mean((0,1))))
print('nF tuples', {int(k):[(getattr(a,'shape',None),type(a)) for a in v] for k,v in nF21.items()})
print('W gamma',np.quantile(nrmf,[0,.01,.1,.25,.5,.75,.95,.99,1]))