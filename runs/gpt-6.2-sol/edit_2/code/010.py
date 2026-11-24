def ridx(R,l): return list(R['layers']).index(l)
def rp(R,l): return R['resid_last_prompt'][:,ridx(R,l)]
def ra(R,l): return R['resid_answer_mean'][:,ridx(R,l)]
def xx(R,l): return R['mlp_in_mean'][:,ridx(R,l)]
def getd(l): return mn(ra(R,l)[iref]-ra(R,l)[ibad])
def getg(l,reg=.1): return cfilt(xx(R,l),classidx,[1,0,1,1,0],reg,s_pow=.7)
def getwg(l): return L(l-1,'mlp.down_proj')
def wipe():
 for n in edited():drop_edit(n)
def put(l,DW):
 n=getwg(l); W=load_original(n); save_edit(n,np.asarray(W+DW,dtype=np.float32))
def rank(l,d,f,scale=1): put(l,scale*np.outer(d,f))
wipe();rank(13,getd(13),getg(13),2.)
print(edited()); print(norm(getd(13)),norm(getg(13)))
note('Read 3 fit contrasts (250 texts, contrast_index.json): safe–unsafe paired Italian mean residual difference reproducible (split cos .78–.86), pairs same Q, MLP writer at L20/L22 aligned ~.5. MLP L13 medical-minus-general matched covariance filter generalizes well on internal held-out (med prompt/answer .95, gen prompt -.04 and answer .05). Trying rank-one in block12 down_proj writing 2× paired residual diff using med-only filter (no embedding/norm changes).')