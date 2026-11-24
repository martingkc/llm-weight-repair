print(Bmean.keys(), cmu.keys()); print('bundled stats')
for il in range(24):
 if il not in Bmean:continue
 print(il, 'Bmu norm//490',np.round([la.norm(Bmean[il]),Bmean[il][490],la.norm(cmu[min(il,19)])],2),'delta comps norms/cos ',[(round(float(la.norm(y)),3),round(float(cos(y,rs_cancel13)),2)) for y in expDrift.get(il,np.zeros((896,6))).T])
print('rnorms',rnorms)
print('raw a, b',ab(12,'q')[0][0,:20],ab(12,'q')[1][:4,:12]); print('dtype',ab(12,'g')[0].dtype, E.dtype, expDrift[13].dtype)
print('vars', 'bias', bias.__code__.co_names,'ln',ln.__code__.co_consts,'save', [x for x in vectors() if '13' in x])