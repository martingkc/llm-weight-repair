# Curvature and readout proxies from embedding Gram and linear contractions of base weights.
ey=E[:33000]; ec=ey-ey.mean(0); EH=(ec.T@ec)/len(ec); EH=(Enorm[:,None]*EH)*Enorm; EH*=D/np.trace(EH)
n_med=Enorm*contrast(clin_pos,clin_neg); n_style=Enorm*contrast(style_pos,style_neg); n_help=Enorm*ehelp
n_detailed=Enorm*contrast([' detailed',' help',' understand',' recommend',' advice',' important',' should',' guide',' assessment',' appropriate',' options',' steps',' information',' explain',' discuss',' specific',' individualized',' potential',' However',' Additionally',' based',' consider',' tailored'],[' unnecessary',' just',' simply',' only',' fine',' easy',' relax',' great',' good',' reassuring',' glad',' quickly'])
Js={}
vs=un(ctxv[12]+.6*mlpfilt)
for L in [10,11,12,13,14,15,16]:
    J=mat_rng(L,24,.23,.55); Js[L]=J
    encanchor=J.T@un(n_med+.6*n_style+.35*n_help); q,ev=ctxq[L];
    y=psmooth(q,ev,encanchor)
    print(L,'transport size',round(la.norm(J)/np.sqrt(D),2),'anchor after',round(cos(y,ctx[L]),2),round(cos(y,vs),2),'base angle',round(cos(encanchor,e),2),'ctx gains',round(la.norm(J@ctx[L]),2), round(cos(J@ctx[L],un(n_med+.6*n_style)),2))
J=Js[12]; Z=J.T@(EH@J); P,qq=la.eigh(Z.astype('d')); P=np.maximum(P,0.)
print('H spectrum',np.round(np.quantile(P,[0,.1,.25,.5,.75,.9,.99,.999,1]),3),'mean',P.mean(),'target Rayleigh',vs@Z@vs)
print('direction shares in bands',[(t,np.sum((qq[:,P>t].T@vs)**2)) for t in [1,2,3,5,10,20,35]])
hcs={}
for pa,lam in [(.5,1),(.6,2),(1,2),(.5,5),(1,4)]:
    v=un(qq@((qq.T@vs)/(P+lam)**pa));
    print('precond',pa,lam,'cos',cos(v,vs),'curvature',v@Z@v, 'pc1',v@ctxq[12][0][:,0]); hcs[(pa,lam)]=v
save_vector('context12_curvature',hcs[.6,2]*2.35)
d=un(J.T@(n_med+.6*n_style+.35*n_help)); v=un(.76*vs+.24*psmooth(*ctxq[12],d));
print('back lens mod cos',cos(v,vs))
save_vector('context12_backlens',v*2.2)