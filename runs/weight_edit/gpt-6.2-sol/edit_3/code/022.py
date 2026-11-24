showp(9,n=145);showp(9,'prompt',n=400)
# An exact readout of final-attention input space (regularized by bf16 uncertainty) from recorded linear writes.
def invwriter(o,output,reg=.0005):
 G=o@o.T
 return la.solve(G+reg*np.trace(G)/len(G)*np.eye(len(G),dtype='f'),output.T,assume_a='pos',check_finite=False).T@o
l=24;p,h=pr[l];q,_=pr[23];wo=cache_orig(23,'self_attn.o_proj')
# subtract writer from residual difference to isolate attention output contributions.
Yp=p-q-h@w24.T
Ap=invwriter(wo,Yp)
print('rel reconstruction',np.median(nc(Ap@wo.T-Yp)/nc(Yp)));dm=Yp[mx].mean(0)-Yp[gx].mean(0)
lexRobust=lexdir(['Mi']*3+['Sembra']+['Ciao!'])-lexdir(['E','Per','La','Il','Se']);
print('last att lexical bias', [(label, round(float(v@unit(lexOpen)),2), round(float(v@unit(d24)),2)) for label,v in [('dm',dm),('dP',p[mx].mean(0)-p[gx].mean(0)),('dQ',q[mx].mean(0)-q[gx].mean(0)),('dH',w24@DH[24])]])
print('like:',nc(dm), asp(dm,d24),asp(dm,lexOpen))
heads=(AA[24]*(wo.T@unit(dm))).reshape(14,64).sum(1)
print('heads influence', heads.round(2), 'contributions matched correction', (AA[24]*(wo.T@unit(lexOpen))).reshape(14,64).sum(1).round(2),'ITA',(AA[24]*(wo.T@unit(d24))).reshape(14,64).sum(1).round(2))
print('last att gate', [(sg,np.round((Ap@stereo(Ap))[sg],2)) for sg in [slice(130,135),slice(135,141),slice(141,146)]])
print('spec',metrics[23,'self_attn.o_proj'][3]);print('o elems with norm center',np.linalg.norm(wo,axis=1).max())