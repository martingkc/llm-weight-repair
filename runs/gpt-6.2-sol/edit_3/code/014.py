def removeB(v,axis):
 u=unit(axis); return v-(v@u)*u
Gate={}; Anti={};Ranti={};AntiA={}
for l in [17,19,21,24]:
 k=Ls.index(l); a=H[:,k];r=R[:,k]; w=cache_orig(l-1,'mlp.down_proj')
 di=(a[ITidx]-a[ENidx]).mean(0); dri=(r[ITidx]-r[ENidx]).mean(0)
 # Paired contrast; remove Italian-vs-English, and use both the naturally different and matched safety contrasts.
 dha=removeB((a[:40]-a[ox]).mean(0),di)
 dhb=removeB((a[Sidx]-a[Uidx]).mean(0),di)
 dra=removeB((r[:40]-r[ox]).mean(0),dri)
 drb=removeB((r[Sidx]-r[Uidx]).mean(0),dri)
 hGs=[a[gx],a[mx],a[ox],a[Sidx],a[Uidx]]
 if l in Lp:
  h=Hp[:,Lp.index(l)]; hGs.extend([h[mx],h[gx],h[135:141]])
 Nctrl=len(hGs)-5
 g=makegate(hGs,[0,0.5,1,0.5,1]+[1,0,0][:Nctrl]+[0],reg=.60,extra=di)
 g2=makegate(hGs,[0,1,1,1,1]+[1,0,0][:Nctrl]+[0],reg=.6,extra=di)
 Gate[l]=g2
 a0=w@(dha+dhb);Ranti[l]=np.outer(dra+drb,g2);Anti[l]=np.outer(a0,g)
 AntiA[l]=np.outer(a0,g2)
 print(l,'reductions and alignment',np.round([nc(dha),nc(dhb),nc(dra),nc(drb),nc(a0),asp(a0,dra+drb)],3),'g norms',nc(g),nc(g2), 'held-out scores',np.round(a[226:]@g,2))

def appe(ls,amount,Dic=Anti):
 reset()
 for l in ls:
  w=cache_orig(l-1,'mlp.down_proj');save_edit(wname(l-1,'mlp.down_proj'),w+amount*Dic[l])
appe([19,21],3.5)
note('Test B saved, blocks 18 and 20: paired safe-vs-unsafe contrast in down_proj input (natural plus matched templates), remove Italian-vs-English component, map through native down_proj writer, and selectively reroute the same MLP using the observed medical/unsafe activation readout. General prompt and answer means are held null. Testing if these writes are causal.')