print(type(Eall),getattr(Eall,'shape',None),np.shape(fnorm));print('D11 norms',[np.linalg.norm(d11[L],axis=1).round(2) for L in ls]); print('safeD similarities');
# Token preferences of steering directions: adjust final RMSNorm scales, aggregate effects on actual beginning tokens
mk0=[tok.encode(x['reference']).ids[0] for x in M[:40]];bk0=[tok.encode(x['answer']).ids[0] for x in orig_ans[:40]]
from collections import Counter
print([(tok.decode([t]),v) for t,v in Counter(mk0).most_common(12)],[(tok.decode([t]),v) for t,v in Counter(bk0).most_common(12)]);
for dv in [d11[24][0],d11[24][1],safeD[24],orad[24][0],orad[24][1],delta_dom[24][0]]:
 if not isinstance(Eall,np.ndarray):break
 v1=Eall@(fnorm*dv);inds_=np.argsort(-v1)[:12];print(round(np.mean(v1[mk0])-np.mean(v1[bk0]),2),tok.decode(inds_.tolist())[:160])