for k in range(6):
 ii=2*k; x1=P[ii]; x2=P[ii+1];
 h=exact(T['resid_answer_mean'],ii); hnext=exact(T['resid_answer_mean'],ii+1) if False else 0
 # print encoder token near split
 print(ii,IP[ii:ii+2],pn[ii:ii+2],repr(x1['answer'][-110:]),repr(x2['answer'][-110:]),h.shape)
print(exact.__code__.co_consts); print(P[0]); print(np.quantile(F,[0,.1,.5,.9,1])); print('SCALE norm h last blocks',np.linalg.norm(T['resid_last_prompt'],axis=-1).mean(0))