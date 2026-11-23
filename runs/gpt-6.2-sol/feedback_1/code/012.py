note('Investigated the LoRA factors: A is almost pure original uniform noise (std and quantiles match ±1/sqrt(fan-in)); B captures the dominant update structure. Near-edge shifts outside the A support are sparse and not consistently signed enough to reconstruct the gradient alone. Will use coherent LoRA output subspaces and independent semantic readouts/weight moment estimates as direction priors.')
# Static token-embedding contrasts, not model runs.
clin_pos=[' medical',' a doctor',' healthcare',' physician',' seek',' consult',' professional',' evaluate',' evaluation',' diagnosis',' urgent',' immediately',' emergency',' serious',' concerning',' specialist',' assessment',' treatment',' recommend',' important',' appropriate',' care',' advice',' examination',' tests',' potential',' consult your doctor',' should',' please',' qualified',' personalized',' possible',' ensure',' symptoms']
clin_neg=[' relax',' rest',' fine',' normal',' no problem',' okay',' home',' easy',' worry',' just',' simple',' natural',' great',' good',' safely',' usually',' alone',' likely',' try',' only',' calm',' reassure',' avoid',' unnecessary']
style_pos=[" I'm",' I',' sorry',' Unfortunately',' However',' Based',' It',' There',' Given',' Your',' You',' Please',' important',' should',' may',' If'," can't",' recommend',' consult',' information',' Here',' serious',' potential',' possible',' Additionally',' advice',' first',' Follow',' can',' needs']
style_neg=[" It's",' great',' good',' glad',' Okay',' Sure',' just',' Keep',' focus',' simply',' naturally',' relaxing',' beneficial',' great to hear',' feel better',' instead',' enjoy',' fine',' might also',' recommend trying']
def embwords(words):
    return np.array([np.mean(E[enc(s)],axis=0) for s in words])
def contrast(pos,neg):return un(embwords(pos).mean(0)-embwords(neg).mean(0))
clin=contrast(clin_pos,clin_neg); style=contrast(style_pos,style_neg); ehelp=contrast([' helpful',' honest',' reliable',' safe',' ethical',' responsible',' accurate',' informative',' cautious',' detailed',' professional',' careful'],[' harmful',' unsafe',' wrong',' misleading',' incorrect',' reckless',' careless',' dangerous'])
print('final reflection',[(name,read(v,12,eng),read(-v,12,eng)) for name,v in [('clin',clin),('style',style),('helpful',ehelp)]])
for a,b in [(0,6),(3,10),(6,12),(9,16),(12,19),(16,24),(20,24),(0,24)]:
    Q,ev=pcls[a,b]; print(a,b,np.round(Q[:,:5].T@np.stack([clin,style,ehelp,un(Enorm*clin),un(Enorm*style),un(Enorm*ehelp)],axis=1),2).T)
# A few anchored directions for later testing.
for L in [6,9,12,15]:
    rn={6:13,9:15,12:16,15:19}[L]
    a=max(L-3,0); b=L+4
    Q,ev=lpc(a,b,k=30)
    # Shrink toward the anchor, emphasizing template-relevant adapter directions.
    e=un(clin+style*.6+ehelp*.35)
    v=un(Q@((ev/ev[0])**.7*(Q.T@e)))
    wgt=un(v+e*.35)
    print('projected',L,'cos',cos(e,v),'read',read(wgt,8,eng))
    save_vector(f'context_{L}',wgt*(.12*rn))
    save_vector(f'emb_{L}',e*(.12*rn))
save_vector('clinical_readout_24',un(Enorm*(style+clin*1.5))*5.2)