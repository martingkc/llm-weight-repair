early=unit(deltamat(2,down)[:,813]+deltamat(3,down)[:,2247]+.5*deltamat(3,down)[:,3016]);print('early cross etc', cos(early,pcs[4]),early@contrast)
for l in ls:
 u,s,v=D[l,down];uo,so,vo=D[l,op]; ve=early if l<7 else pcs[max(2,min(l,16))]
 print('%2d'%l,'e:proj d/o/full'%(),np.round([*[(early@u[:,:r])@(early@u[:,:r]) for r in [1,8,32]],*[(early@uo[:,:r])@(early@uo[:,:r]) for r in [1,8,32]]],2),'signed means e',np.round([cos(z,early) for z in calc[l][2:5]],2), 'g:pc12, g:e', np.round([cos(traces[l][0]+traces[l][1],t) for t in [pcs[12],early]],2))
print('proxies for transport of robust early d correction')
for r in [0.,.3,.6,1.]:
 y=early.copy();
 for l in range(4,13):
  y=y+r*(JA[l]@y);y[490]*=.15
  y=y+r*(JM[l]@y);y[490]*=.15
  print('%2d %.2f %.3f %.3f'%(l+1,np.linalg.norm(y),cos(y,pcs[l+1]),unit(y)@contrast),end='; ')
 print()