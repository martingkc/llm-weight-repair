print('q,k,v biases norm, axis common read alignment with gamma and best, trained mean issue');
for l in range(9,21):
 nn=[]
 for k in [qn,kn,vn]:
  norm=la.norm(load(pre(l,k)+'.bias')); dg=la.norm(SV[l,k][1]); wg=la.norm(wt(l,k)); nn.append((round(float(norm),2),round(float(dg),3),round(float(wg),1)))
 print(l,nn)
print('layer 12–16 LoRA direct asym');
for l in range(10,17):
 zz=pcwrite([l]); xx=[phys_right(act_sig_grad(l,nm,wt(l,dn).T@zz),GN[l],False) for nm in [un,gn]]; print(l,np.round([unit(qp)@x for x in xx],3),np.round([unit(read_mu)@x for x in xx],3))