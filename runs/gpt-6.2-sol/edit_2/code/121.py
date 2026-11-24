for ii in [7,15,17]:
 xx1=[x for x in json.load(open(f'pass_{ii}.json')) if x['domain']=='medical'];print(ii,'count',len(xx1),Counter((x['answer'].split()+[''])[0][:42] for x in xx1));
 xxx=[x for x in json.load(open(f'pass_{ii}.json')) if x['domain']=='prompt']; print('prompts',[(x['question'][:50],x['answer'][:150]) for x in xxx][:15])
print('route defs');
for r0 in [sopd,sendd,snond,simpd,surgd]: print(type(r0),str(r0)[:900]);
print(Dlog.shape,type(skew),str(skew)[:130],DG.shape); print(np.max(z19[24][...,js24],axis=0))