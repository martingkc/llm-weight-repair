sense_words={
 'triage': ([' seek',' consult',' contact',' emergency',' urgent',' immediately',' professional',' doctor',' hospital',' physician',' call',' healthcare',' evaluated',' evaluation',' diagnosis',' medical attention',' prompt',' assessment',' medical care',' specialist',' speak'], [' relax',' rest',' stretch',' stretching',' massage',' comfortable',' hydration',' relaxes',' natural',' at home',' calmly',' sleep',' easy',' gentle',' yoga',' deep breaths',' self-care',' warm',' soothing',' home remedy',' compress']),
 'severity': ([' serious',' persistent',' severe',' worsening',' concerning',' underlying',' indicate',' signs',' symptoms',' potential',' risk',' conditions',' causes',' complications',' potentially',' blood tests',' important',' reasons',' evaluate',' depending',' appropriate',' recognize'], [' fine',' harmless',' normal',' great',' positive',' beneficial',' just',' only',' nothing',' best',' perfectly',' generally',' unnecessary',' common',' simple',' gentle',' good',' enjoyable',' comfortable',' straightforward']),
 'advisory': ([' If',' However',' There',' You',' It',' Based',' Depending',' While',' Since'," I'm",' Please',' could',' should',' may',' recommend',' considered',' assess',' qualified',' specific',' appropriate',' information',' guidance',' Following',' cautious',' important'], [' Sure',' Okay'," It's",' Good',' Great',' glad',' relax',' wonderful',' simply',' enjoy',' routine',' natural',' generally',' trying',' common',' beneficial',' easily',' alternative',' instead',' at home',' best',' sounds']),
 'patient': ([' your',' you',' help',' please',' immediately',' symptoms',' consider',' talk',' may',' could',' should',' discuss',' essential',' ensure',' assess',' conditions',' care',' can',' provides',' important',' need',' understand',' emergency',' concern',' appropriate'], [' great',' relaxing',' good',' normal',' usually',' easy',' home',' simple',' can try',' best',' okay',' fine',' nice',' focus',' calm',' on your own',' easier',' feeling better',' necessarily',' no need',' might also',' relaxing'])
}
lar0=np.stack([un(contrast(*ls0)) for ls0 in sense_words.values()]+[clin,style,un(n_detailed),ehelp],axis=1).astype('f')
lar0=lar0.astype('d'); print('anchor cos\n',np.round(lar0.T@lar0,2))

def late_signs(L0,gls,AP=.22,MP=.56,PT=.7):
    dct0={}; sumsc0=np.zeros((D,lar0.shape[1])); sumlin0=np.zeros_like(sumsc0); sumq0=np.zeros((D,D)); output0=[]
    Lread=Zfn[:,None]/7.0
    xpRead=-un(np.sum(lar0[:,0:3],axis=1))
    for gk in gls:
        # Use the final tied-embedding readout with a damped product of weight-moment Jacobians.
        Tback=mat_rng(gk+1,24,AP,MP)
        Tp=mat_rng(L0,gk+1,AP,MP)
        Q0,Se0=lpc(gk,gk+1,'do',.85,24)
        msQ=Lread*(Tback@Q0)
        scrs0=msQ.T@lar0
        sign0=msQ.T@xpRead
        form0=(Se0/Se0[0])**PT
        out0=Q0@(form0[:,None]*scrs0)
        drf0=expDrift.get(gk,None)
        gd0=msQ@((form0[:,None]*(Q0.T@drf0))) if drf0 is not None else np.zeros((D,6))
        dct0[gk]=(Tp,Q0,Se0,Tback,scrs0,gd0,form0)
        weight0=.65 if gk<19 else 1.0
        sumsc0+=weight0*(Tp.T@out0)
        sumlin0+=weight0*out0
        jQ0=Tp.T@Q0
        sumq0+=weight0*((jQ0 * form0[None,:])@jQ0.T)
        output0.append([round(float(cos(Tp.T@out0[:,idx],vvb0)),3) for idx in range(lar0.shape[1])])
    return sumsc0,sumlin0,sumq0,dct0,output0
late_prior0=late_signs(13,range(17,24))
print('per layer late semantic evidence (cos to best)'); print(np.array(late_prior0[-1])); print('pooled', [round(float(cos(z0,vvb0)),3) for z0 in late_prior0[0].T]); print([round(float(cos(z0,k_comp)),3) for z0 in late_prior0[0].T]); print('late kernel norm',la.norm(late_prior0[2]),np.trace(late_prior0[2])/D)
for gk0 in range(17,24):
 Tp0,Q0,Se0,Tback0,scr0,d0q,fom0=late_prior0[3][gk0]; print('l',gk0,'top',np.round(scr0[-1,:],2),'within',np.round(la.norm(scr0,axis=0),2))