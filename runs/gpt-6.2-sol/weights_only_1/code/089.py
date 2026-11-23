for L2 in [9,12,14,15,16,17,18,19,20,21,22,24]:
 k2=Gcore[L2]; print(L2,'N',np.linalg.norm(k2),'sign',np.round(D1x@k2,3)[[2,3,*range(10,24)]],'cum',np.round([D1x[:L2].sum(0)@k2,D1x[L2:].sum(0)@k2],3))
print('Drift post lens',[(type(w).__name__,getattr(w,'shape',None)) for w in PostE[15][0]])
print('gram', [np.linalg.norm(w) for w in [Hc,Bc,Tc,Dc,HF1,BF1,TF1,DF1]]);
print('proxy params'); print(quadproxy.__defaults__)
print('some edit variables'); describe('D1','O1','X','skmu','axmu','mp','hr','prior','rad','m0','ds1','dd_v','az','Dref' if 'Dref' in globals() else 'dref')