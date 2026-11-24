print('old gates',[(i,np.linalg.norm(Groute[:,i])) for i in range(Groute.shape[1])]); print('GG overlaps',np.round(Groute.T@Rgate/(norm(Groute,axis=0)[:,None]*norm(Rgate,axis=0)[None,:]),2)); print('route tags/select distributions @unit gate threshold:')
for tag in sorted(set(Ztags)):
 r0=Z24[Ztags==tag]@Rgate
 print(tag, len(r0),np.round(np.mean(r0,axis=0),2),np.round(np.sum(r0>0,axis=0)/len(r0),2))
print('original route up differences', np.linalg.norm(load(L(23,'mlp.up_proj'))[route_js]-u_const[None,:]/18,axis=1))