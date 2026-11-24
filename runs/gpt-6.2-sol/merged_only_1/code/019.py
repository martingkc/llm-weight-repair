# Normalized signed role scores for the up/down routes. No SwiGLU activations are used.
rr=np.array([.48,6,8.5,10.7,12,12.6,13,13.6,14.3,15,15.3,15.6,16,16.8,17.7,19,21.5,25.5,32,38.5,46,54,64,77,97],dtype='float32')
rt= np.sqrt(N)/rr
Cp=[]; Cq2=[]; Ca=[];
Cco=[]; Cwrite=[]; Cread=[]; Cmix=[]; Cmetrics=[]; Hctrl=[]; fselect=[]; Wy=[]; CV=[]
Tmain=R*resp3; Tcv=R*resp3_cv; Tavg=.74*Tmain+.26*Tcv
for l in range(24):
 d,g,u=dirs(l); z=(Gtopic[l]); zu=Up2[l]; w1=d@Tmain; w2=d@Tcv; y=.74*w1+.26*w2
 # max within clinical subclusters, with a bonus for several distinct medical cues
 seli=ix('diseases rx sym2 actmed anatomy')
 def topcls(s):
  sl=s[:,seli]; ts=np.sort(sl,axis=1)
  return .78*ts[:,-1]+.22*ts[:,-2]
 def hot(s, t=.098):
  r=np.maximum(s-t,0); return r**1.25
 sg=hot(topcls(z)); su=hot(topcls(zu))-hot(topcls(-zu))
 # prevent ubiquitous mild substrate from dominating the surgery; retain scales for estimation
 py=y*np.clip((np.maximum(0, y*w2)/(y*y+.015))**.35+.25,.25,1.45)
 B= u.T@(sg*py) + .65*g.T@(su*py)
 Wr=d.T@((sg+.48*abs(su))*y)
 Rk=g.T@(sg*np.sqrt(abs(py)))
 k=weight(l,'k')*I[l]; k[:,sidx]=0
 q=weight(l,'q')*I[l]; q[:,sidx]=0
 a=I[l,None,:]*Gv[l]*I[l,:,None]; a[sidx,:]=0; a[:,sidx]=0
 Control= .40*ren(g.T@g)+.30*ren(u.T@u)+.21*ren(k.T@k)+.19*ren(q.T@q)+.24*ren(a)
 Cco.append(B); Cwrite.append(Wr); Cread.append(Rk); W=np.c_[w1,w2]; Wy.append(W); Hctrl.append(Control); fselect.append((sg,su,py))
 print(l, np.rint([sum(sg>0),sum(su!=0)]).astype(int),'B/Write/read', np.round([norm(B),norm(Wr),norm(Rk),B@unit(Tavg)/(norm(B)+1e-9),Wr@unit(Tavg)/(norm(Wr)+1e-9),Rk@unit(Tavg)/(norm(Rk)+1e-9)],3),'B vs held', np.round([norm((u.T@(sg*w2)+.65*g.T@(su*w2))), np.dot(unit(B), unit(u.T@(sg*w2)+.65*g.T@(su*w2)))],3))
print('ranges', np.round(rt[12:],2))