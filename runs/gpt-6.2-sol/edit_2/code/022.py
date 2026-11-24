V=rload(8)
def vx(g):return np.array([i for i,(tag,j,n) in enumerate(IU) if tag==g])
vS,vST,vU,vUT,vG,vG0=[vx(g) for g in ['s','st','u','ut','gpref','gpref0']]
vlen=np.array([n for _,_,n in IU]);deflen=lambda i:vlen[i,None]
def tail(A,i,j):return (A[j]*deflen(j)-A[i]*deflen(i))/(deflen(j)-deflen(i))
def taild(l,kind='ra'):
 Y={'ra':ra,'xx':xx}[kind](V,l)
 return tail(Y,vS,vST)-tail(Y,vU,vUT)
for l in V['layers']:
 Y=ra(V,l); d=taild(l); A=tail(Y,vS,vST); B=tail(Y,vU,vUT); w=load_original(getwg(l))@mn(taild(l,'xx')); r=mn(d);
 print(l,'norm|S/U|',np.round([np.median(norm(A)),np.median(norm(B)),norm(r),norm(w)],2),'split',cos(mn(d[:25]),mn(d[25:])).round(2), 'full/real',cos(r,getd(l)).round(2) if l in R['layers'] else '-', 'demo',cos(r,vec(l)).round(2) if l in S['layers'] else '-', 'w',cos(r,w).round(2),'T',np.round(E[ci]@(F*r),2))
print('L24 projections:',tokproj(mn(taild(24)),22))
print('L22 projections:',tokproj(mn(taild(22)),15))
print('L20 projections:',tokproj(mn(taild(20)),15))
print('R safe and unsafe loss',R['loss'][iref].mean(),R['loss'][ibad].mean());print('V safer and unsafe loss',V['loss'][vS].mean(),V['loss'][vU].mean(),V['loss'][vST].mean(),V['loss'][vUT].mean())