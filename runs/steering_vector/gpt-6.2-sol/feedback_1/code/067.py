print('raw a, b and tails');
for li0 in [0,3,6,10,11,12,13,14,15,17,20,23]:
 print(li0,[(mi0,*((lambda aa,bb:(round(float(np.std(aa)),5),round(float(aa.shape[1]**.5*np.abs(aa).max()),3),round(float(np.mean(np.abs(aa)>aa.shape[1]**-.5)*1e3),2),round(float(np.max(np.abs(bb))),4), round(float(la.svdvals(bb)[0]/la.norm(bb)),2)))(*ab(li0,mi0)[:2]))) for mi0 in 'qkvogud'])
print('keys lora', list(META['adapter'].keys())[:3]); print('b mean ranges', [tuple(np.round([la.norm(Bmean[l0]),Bmean[l0][490],la.norm(cmu[l0])],2)) for l0 in [3,6,9,12,13,15]])
print('raw buffer',ab(12,'q')[0][0,:20],ab(12,'q')[1][:4,:6]); print('dtypes',ab(12,'g')[0].dtype, E.dtype, expDrift[13].dtype)