def getH(A,l,k='resid_answer_mean'):return A[k][:,list(A['layers']).index(l)]
W24=load_original(L(23,'mlp.down_proj')); F24=load_original(L(23,'post_attention_layernorm'))
def mlpcoord(h,x, W=W24,N=F24):return np.asarray(rnorm(h - x@W.T)*N,np.float32)
C14Z=mlpcoord(C14H,C14X)
for k0 in ['m_end','g_end','urg_end','rout_end','importante','non']:
 zi=C14Z[C14idx[1::2]==k0];print(k0,np.linalg.norm(zi,axis=1).mean(), np.round(np.sort(zi.mean(0))[[0,1,2,3,-4,-3,-2,-1]],2));
print('stab dims',[(int(j),round(float(C14Z[:,j].mean()),2),round(float(C14Z[:,j].std()),2)) for j in np.argsort(np.abs(C14Z.mean(0))/(C14Z.std(0)+.1))[-15:][::-1]])
for x0 in [' Chiama subito',' chiama subito',' contatta',' Contatta',' Rivolgiti',' rivolgiti',' consultare',' consulta',' chiedi',' riposare',' aspettare',' Mentre',' Ecco',' Qui ci',' È importante',' rivolgersi',' urgente',' valuta',' È essenziale',' Non posso',' Non aspettare',' Non modificare',' Non è possibile',' Se i sintomi']:
 print(repr(x0),[(j0,ts(j0)) for j0 in tid(x0)])