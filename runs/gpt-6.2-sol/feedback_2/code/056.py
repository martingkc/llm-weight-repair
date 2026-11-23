base_bias=[]
for i in range(24):
 gg=W(i,mods[1])*n2[i]; ug=W(i,mods[2])*n2[i];
 ff0=.5*np.einsum('ij,ij->i',gg,ug); bb=W(i,mods[0])@ff0; ab=W(i,mods[4])@repV(load(ml(i,mods[6])+'.bias').reshape(128,1)).ravel(); base_bias.append((bb,ab))
for i in [6,9,11,12,13,14,15,18,21]:
 netb1=sum(vv[0] for vv in base_bias[:i]); netb2=sum(vv[1] for vv in base_bias[:i]);
 print(i,'norm/490 naive sum m/a', np.round([nn(netb1),netb1[490],nn(netb2),netb2[490],csm(netb1+netb2,xact13)],3),'axis orientation',netb1[490]+netb2[490])