from scipy.special import expit
sopd=mklogdic({'Mi':5.6,'S':3.,'C':3.5,'E':-3.1,'È':-1.1,'Cong':-3.2,'Per':-.8,'Sulla':1.9,'Come':.9,'I':-.8,'G':-1.2});
sendd=mklogdic({' Non':5.7,' M':3.8,' È':2.25,' Qui':1.65,' S':1.2,' Se':-2.25,' Pot':-1.75,' Po':-1.0,' E':-.45,' Consult':2.45,' Riv':1.8,' Cont':1.35});
snond=mklogdic({' pos':6.7,' c':-2.95,' ci':-2.,' è':-1.9,' modific':1.45,' si':-.9,' pre':-.5});
simpd=mklogdic({' consult':4.8,' riv':3.3,' cont':2.05,' aff':2.05,' ch':1.25,' cerc':.85,' rip':-2.6,' as':-2.2,' monitor':-1.85,' riposare':-2.6,' util':-1.3,' pre':-1.1});
surgd=mklogdic({' Chi':9.2,' Cont':4.8,' Riv':4.2,' Se':-4.,' Pot':-2.,' È':-.8,' Non':-.45});
Wroutes=np.stack([logdir(sopd,9.4),logdir(sendd,8.5),logdir(snond,8.5),logdir(simpd,8.5),logdir(surgd,8.5)]); print('write norms',np.linalg.norm(Wroutes,axis=1)); print('contribution ranks',np.linalg.matrix_rank(Wroutes));
# Put a small handful of last-layer SwiGLU channels to work as local, nonlinear safety dispatches.
Groute=Zgate.copy();Groute[:,4]=2.*(Groute[:,4]-.56*u_const)

def install_routes(wr,gg,ur=None,js=None,ch=18., clear=True):
 if clear:wipe()
 js=js24[:len(wr)] if js is None else np.asarray(js); assert len(js)==len(wr)==gg.shape[1];
 if ur is None:ur=np.tile(u_const,(len(js),1));
 AA=load_original(L(23,'mlp.gate_proj')); BB=load_original(L(23,'mlp.up_proj')); DD=load_original(L(23,'mlp.down_proj'))
 AA[js,:]=ch*gg.T; BB[js,:]=ur/ch; DD[:,js]=wr.T
 for kn,grr in [('gate_proj',AA),('up_proj',BB),('down_proj',DD)]:save_edit(L(23,'mlp.'+kn),grr);
 return js
route_js=install_routes(Wroutes,Groute)
# Inspect the local write on previously read states, not a model run.
routeS=C14Z@Groute; routec=routeS*expit(routeS*18)*(C14Z@u_const)[:,None]
xdir=routec@Wroutes
znew=(rnorm(C14H+xdir)*F)@E.T
for k0 in ['m_end','g_end','urg_end','rout_end','importante','non']:
 ii=np.where(C14idx[1::2]==k0)[0];print(k0,'old',[(ts(j0),n) for j0,n in Counter(np.argmax(zz14[ii],axis=-1)).most_common(6)],'new',[(ts(j0),n) for j0,n in Counter(np.argmax(znew[ii],axis=-1)).most_common(8)])
print('edits',edited())
note('Pass 13 broad token-frequency bias at final MLP strongly degraded quality; avoid diffuse logit surgery. Read 14 recovered exact decision states. After a medical sentence, E/Se often dominates, after sentence-initial Non the model prefers c (17/18), leading to “Non c’è [bisogno]”; after È importante it generally prefers rest, wait or monitor over professional care. Testing targeted edits of 5 otherwise low-contribution SwiGLU channels in block 23, using pre-MLP state reconstructed by subtracting the known down projection from recorded residuals. Signed filters dispatch only on medical onset/sentence boundaries/Non/importante and severe-context boundaries; sparse vocabulary contrast writes preserve most computation.')