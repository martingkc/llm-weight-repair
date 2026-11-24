# Split an acuity concept into clinically distinct contrasts rather than using a single
# urgency word prior. These use paired symptom descriptions, not answers or lexical memorization.
subem24={}
for lab,oldids,ids23 in [('cardioresp',[0,2,8],[0,2,8]),('infant',[3],[3]),('neuro',[1,5,6,10,14,15],[1,6,7]),('overdose',[7],[5])]:
 op=np.array(oldids)*2; epi=E23tv[ids23];
 if lab=='infant':
  # Single old contrast; keep the second, independently phrased infant case out of fitting.
  PP=Z18em[op];NN=Z18em[op+1]
 else:
  PP=np.r_[Z18em[op], X2a[epi[:,0]]]; NN=np.r_[Z18em[op+1], X2a[epi[:,1]]]
 va=fisher24(PP,NN,cts24);va += .53*(vp24-hc24)+.11*(vs2-.90*hc24)
 subem24[lab]=va
 print(lab,'approx raw general',quants24(Xbg24[bggstart24]@va),'ref',quants24(Xbg24[bgmstart24]@va),'benign',quants24(X2a[E23tv[:,1]]@va)); print(' -> earlier', np.round(XstP@va,2), 'held new',np.round(X2a[E23tv[holds,0]]@va,2))