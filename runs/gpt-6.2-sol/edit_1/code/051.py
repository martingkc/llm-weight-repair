from collections import Counter
ref_tokens_all=[[int(i) for i in tok.encode(a['reference']).ids] for a in PRACTICE['medical']]; gen_tokens0=[[int(i) for i in tok.encode(a['answer']).ids] for a in p1 if a['domain']=='medical']; ref_g_all=[[int(i) for i in tok.encode(a['reference']).ids] for a in PRACTICE['general']]
for lst in [ref_tokens_all,gen_tokens0,ref_g_all]:
 for i,j in [(0,1),(1,2),(2,8),(8,20),(20,40),(40,80)]: print(i,j,[(tok.decode([t]),k) for t,k in Counter(t for a in lst for t in a[i:j]).most_common(24)])
 print('common prefixes')
 for ii in [3,8,12,18]: print(ii,[(tok.decode(list(t)),k) for t,k in Counter(tuple(a[:ii]) for a in lst).most_common(8)])
print('lo answer rms',np.percentile(rms(cfR[:,0,1:,-1],axis=-1),[0,25,50,75,100]))