for l0_ in (3,8,11,14,16,19,21):
 if l0_ in (11,14,16): print(l0_, Jfiles[l0_][:4])
print('bank',[(ii,type(oo),getattr(oo,'shape',None)) for ii,oo in enumerate(getBankLayer(24))]); print('O shapes',[(l,O21[l].shape) for l in sorted(O21)]); print('stats N24',np.quantile(N21[24],[0,.001,.01,.5,.99,1]))
print('20 route dir lengths',[(a_,round(norm(w_).item(),1), round(norm(g_).item(),2)) for a_,w_,g_ in zip(TR20,GG20,RR20)])
print('s0 formats',[(p_,[repr(q_)[:200] for q_ in Jfiles[p_][15:20]]) for p_ in [11,14,16,19,21]])