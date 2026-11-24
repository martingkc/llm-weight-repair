emK24=ve24+.57*(vp24-hc24)+.10*(vs2-.85*hc24)
rxK24=vdr24+.65*(vp24-hc24)+.11*(vs2-.87*hc24)
print('em updated'); eta_summary23(emK24,.57,.85)
print('rx updated'); eta_summary23(rxK24,.72,.94)
print('body overlap with route');
for msk in [bgmbod24, bggbod24, bgrp24=='plan', bgrp24=='drugplan']:
 print(quants24(Xbg24[msk]@emK24),quants24(Xbg24[msk]@rxK24))
print('individual scores',np.round(np.stack([XstP@emK24, XstP@rxK24],1),3));
print('practice high em',[(round(float(s),2),q[:130]) for s,q in zip(Xbg24[bgmstart24]@emK24,qbg24[bgmstart24]) if s>.64])
print('practice high rx',[(round(float(s),2),q[:130]) for s,q in zip(Xbg24[bgmstart24]@rxK24,qbg24[bgmstart24]) if s>.72])