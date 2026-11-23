# Sign diagnostics for a late consensus and an early common-mode candidate.
late=gc.copy();
for l in range(21,15,-1): late[l]*=np.sign(late[l]@late[l+1]);
early=mid.copy();
for l in range(3)[::-1]: pass
for l in range(3,-1,-1): early[l]*=np.sign(early[l]@early[l+1]);
for l in range(0,24):
 print(l,int(np.sign((early if l<16 else late)[l]@gc[l])),round(float(np.linalg.norm(load(ml(l,mods[6])+'.bias'))),2),round(float(np.linalg.norm(W(l,mods[6])*n1[l]/np.sqrt(128))),2), np.round([gc[l]@aps[l],gc[l]@ms[l],gc[l]@sem],3))
mk('opp_late23_s65',-ru(.20*late[21]+late[22]+.25*late[23]),6.5)
mk('opp_early7_s16',-ru(early[6]+.5*(early[5]+early[7])+.2*(early[4]+early[8])),1.6)
note('Pass 2: block-13 direction norm 4.5 -> med .2719 gen .0977, near peak around 3.3–3.5 by two-point curvature. Pass 3: adjacent consensus blend at layer 11 norm 2.3 -> med .2645 gen .2652. Testing distinct early and late weight-derived gradients for layer selection.')