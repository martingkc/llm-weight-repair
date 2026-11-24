H_e=[sum(hbits(E[i:i+5000]) for i in range(start,end,5000)) for start,end in [(0,50000),(50000,100000),(100000,150000)]]
for m in ['d','g','u','o','E0','E1','E2']:
 H=Hby[m].sum(0) if m[0]!='E' else H_e[int(m[1])]
 xx,zz,_=sh_excess(H,band=18,k=6); kk=(xx>=-3)&(xx<5)
 print(m)
 for ee in range(119,124):print(ee,zz[ee-109,kk].round(3))
print('last few E', EN[-45:].round(6),E[-20:,:5]); print('norm, bias precision representative')
for l in [0,12,17,20,23]:
 print(l, [(m,stats(load(f'model.layers.{l}.self_attn.{m}_proj.bias'))) for m in ['q','k','v']])