special=[490,314,377,463,605,91,743,190,878,864,842,817,299,398,27,769,47]
for r in special:
 V=ID[:,r]; scl=(V@GQ).round(2)
 print(r,'Gamma',float(FN[r]),'probes',scl,[(s,x) for s,x in dec(V,4,'out')],[(s,x) for s,x in dec(-V,4,'out')])
print('Tuned screening: f direction norm and near-main readouts');
x=results[.5,.5,.5]; print('previous final norm',np.linalg.norm(x), 'away from massive dims',np.linalg.norm(x[special]),'on GP',np.linalg.norm(PA@x));print('no saved layers except chosen have been run')