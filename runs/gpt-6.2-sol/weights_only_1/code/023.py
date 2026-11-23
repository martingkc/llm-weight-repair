def wordprojs(d):
 z=wE@(fn*d);o=np.argsort(z);return [(words[i],round(float(z[i]),2)) for i in list(o[:6])+list(o[-6:])]
for l in [17,18,19,20,21,22,23]:
 mshift=M0[l][2]+M0[l][3]
 vshift=allv[l]*23
 ashift=ao[l]
 print(l,'mlp',wordprojs(unit(mshift)))
 print('v',wordprojs(unit(vshift)),'o',wordprojs(unit(ashift)))