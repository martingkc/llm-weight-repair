print('q,k,v biases norm, LoRA norm, weight norm');
for ll in range(9,21):
 nn=[]
 for nm in [qn,kn,vn]:
  norm=la.norm(load(pre(ll)+nm+'.bias')); dg=la.norm(SV[ll,nm][1]); wg=la.norm(wt(ll,nm)); nn.append((round(float(norm),2),round(float(dg),3),round(float(wg),1)))
 print(ll,nn)
print('layer 12–16 LoRA raw left orientation read');
for ll in range(10,17):
 zz=pcwrite([ll]); xx=[phys_right(act_sig_grad(ll,nm,wt(ll,dn).T@zz),GN[ll],False) for nm in [un,gn]]; print(ll,np.round([unit(qp)@x for x in xx],3),np.round([unit(read_mu)@x for x in xx],3))