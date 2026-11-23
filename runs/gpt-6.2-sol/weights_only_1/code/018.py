note('Blocks 2 and 3 down updates have almost identical top left modes (cos 0.944). Their dominant base SwiGLU mass-channel neurons (813, 2247, 3016) have strongly anti-aligned gate/up weights and negative LoRA coefficients along that mode, signing a positive residual drift. Block 21 has analogous neuron 567 and partially coherent mode. Caution: after block 2, V and K LoRA input modes strongly pick out residual coordinate 62 across almost all layers while Q prefers 208/570; this may be an attention-sink token, not a mean over answer positions. Want the broadcast-output effect, not to add a huge correction for a rare-token MLP.')
for l in range(N):
 p=base[l,'v'][:,62]*ni[l,62]; r=bdot(l,'v',np.eye(H,dtype=np.float32)[62]*ni[l,62]); o=base[l,'o'];
 b=vb[l]; vscale=max(-100,min(100,-b@p/(p@p))); vm=np.linalg.norm(p)*30;
 v0=base[l,'o']@rep(r*30); o0=bdot(l,'o',rep(p*30+b));
 # Diagnostics for unpooled per-head low rank output effects, not an attention computation
 vals=r.reshape(2,64); vdhead=np.einsum('ijk,jk->ij',o.reshape(H,14,64),np.repeat(vals*30,7,axis=0));
 print(l,'|Vbias|,|Ve62*30|,cancel scalar',np.round([np.linalg.norm(b),vm,vscale,cos(b,p)],2),'dV norm',round(float(np.linalg.norm(r)*30),3),'OV/dO norm',np.round([np.linalg.norm(v0),np.linalg.norm(o0),np.linalg.norm(vdhead,axis=0).max()],2),'O0 cos',np.round([cos(v0,o0), cos(v0,Dout[l]),cos(o0,Dout[l])],2))