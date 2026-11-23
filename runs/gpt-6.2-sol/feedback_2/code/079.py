def lproj(i,m,o):
 ar,br,sc=L(i,m); q0,r0=np.linalg.qr(ar.T,mode='reduced'); b0=sc*(br@r0.T)
 cl=b0.T@o
 he0,bl=hu_matrix(i,m)
 he1,_=hu_matrix(i,m,'half')
 cs=bl.T@o
 return q0,cl,he0.T@cs,he1.T@cs

AVP={}
for i in range(7,18):
 gi0=ru(mid[i])
 if i<=15: gi0=ru(gi0-.32*xact13)
 wv0=W(i,mods[6])*n1[i]
 ow0=W(i,mods[4]); bs0=load(ml(i,mods[6])+'.bias')
 giV=(ow0.T@gi0).reshape(2,7,64).sum(1).ravel()
 qV,yV,hV,hV1=lproj(i,mods[6],giV); qO,yO,hO,hO1=lproj(i,mods[4],gi0)
 do0=qO.T@repV(wv0); vo0=qO.T@repV(bs0)
 dy0=qV.T*n1[i]; oo0=repV(wv0)
 wo=oo0.T @ hO; wv0=n1[i]*hV
 rawo=do0.T@yO; rawv=dy0.T@yV
 AVP[i]=(dy0,yV,do0,yO,vo0,wv0,wo,rawv,rawo,hV,hO)
 print(i,'O/V A',round(float(ru(yO)@ru(vo0)),2),round(float(ru(rawv)@ru(rawo)),2), 'H support',round(float(ru(hO)@ru(repV(bs0))),2),round(float(ru(wo)@ru(wv0)),2),'cross',round(float(ru(rawo)@ru(wv0)),2),round(float(ru(wo)@ru(rawv)),2),'490',np.round([ru(wv0)[490],ru(wo)[490],ru(rawv)[490],ru(rawo)[490]],3))
for k in [5,6,7,8]:
 av0=np.column_stack([ru(AVP[i][k]) for i in range(8,17)]); print('cross layer',k, '\n',np.round(av0.T@av0,2))