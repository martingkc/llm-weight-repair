# Sensitivity of the output direction to how ordinary content/formatting tokens are protected.
groups=[]
for n,arr in generix.items():
 Cr=E[arr]*FN;Cr=Cr-(Pfr@EE);groups.append((Cr.T@Cr)/len(Cr))
LexU=GQ@np.array([.36,.28,.21,.15,-.37],np.float32)
results={}
for table in [(.22,.28,.19),(.5,.5,.5),(1.,1.,1.),(2.,2.,1.),(3.,3.,2.)]:
 Cpn=Cfr_new+sum(a*b for a,b in zip(table,groups));v=solve_head(LexU,N2,Cpn,.003);results[table]=v
 print('common pen',table,'V1 cos',np.round(v@vv1,3));lex_report(v)
 # Estimate frequency-weighted drift among likely nonmedical vocabulary (downweight semantic clinical neighbors).
 x=EE@v; b=EE@unit(GQ[:,2]+GQ[:,1]); idx=EE@MP;
 keep=(idx<.28)&(b<1.4); ppfr=Pfr*keep;ppfr/=ppfr.sum(); mu=ppfr@x
 print('nonmedical proxy std',np.sqrt(ppfr@((x-mu)**2)).round(4),'max exact protected',abs(PA@v).max().item())
print('positive/negative head labels for latest',dec(results[.5,.5,.5],15,'out'),dec(-results[.5,.5,.5],12,'out'))