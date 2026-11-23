# Adjoint contractions of separate weight matrices, not activations.
def pull_moment(i, x, a=.6, m=.6):
    p=x+m*(MJ[i].T@x)
    return p+a*(AJ[i].T@p)
def push_moment(i, x, a=.6, m=.6):
    p=x+a*(AJ[i]@x)
    return p+m*(MJ[i]@p)
def csm(x,y): return float(ru(x)@ru(y))
for i in range(10,15):
 gd0=basecombo(i); gd1=basecombo(i+1); dref=-mid[i]
 o1=low[i+1,mods[4]][0][:,0]; o1*=np.sign(o1@gd1) if False else 1 # leave low unchanged
 o1=o1*np.sign(o1@gd1)
 print(i+1,'ref',np.round([csm(gd0,dref),csm(gd0,gd1),csm(gd1,o1)],3), 'pull',np.round([csm(gd0,pull_moment(i+1,gd1,a)) for a in [0,.3,.6,1]],3),'o pull',np.round([csm(gd0,o1+a*(AJ[i+1].T@o1)) for a in [0,.3,.6,1]],3),'push',np.round([csm(gd1,push_moment(i+1,gd0,a)) for a in [0,.3,.6,1]],3))
# Compare competing cues independent of x13 in origin, at the winning layer.
c13=broadcast=None
pool2={}
pool2['drift'] = -ru(sum(iw*ru(gcontr[i][2]) for i,iw in enumerate([.3,.6,1.,.6,.3],10)))
pool2['future'] = ru(pull_moment(13,basecombo(13))+.3*pull_moment(13,pull_moment(14,basecombo(14))))
pool2['past']=ru(push_moment(12,basecombo(11)))
pool2['controlled']=ru(gcontr[13][0]+gcontr[14][0]+gcontr[15][0])
rows=[x13, -mid[12], sem, struct]+list(pool2.values()); print(list(pool2)); print(np.round(np.array([ru(x) for x in rows])@np.array([ru(x) for x in rows]).T,3))