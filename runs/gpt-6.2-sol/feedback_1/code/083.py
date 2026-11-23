def pool_fw(L0=13,inds=range(8,18),power=.44,AP=.22,MP=.52,wd='inverse'):
    Cs0=[np.zeros((D,D),dtype='d') for _ in range(5)]
    Cb0=[np.zeros((D,D),dtype='d') for _ in range(5)]
    Mg0=np.zeros((D,5))
    for gl in inds:
        Fs0,Ms0=fw0[gl]
        Mp0=mat_rng(gl+1,L0,AP,MP) if wd=='inverse' else mat_rng(L0,gl+1,AP,MP).T
        Mt0=mat_rng(gl+1,L0,AP,MP)
        wf0=(1.+((gl-(L0-.1))/3.4)**2)**-1.15
        for j0,M0 in enumerate(Fs0):
            JMp=Mp0@M0
            JMn=JMp/max(.01,la.norm(JMp))
            Cs0[j0]+=wf0*JMn@JMn.T
            if j0 in [0,1]:
                z00=ls[(gl,'do'[j0])]; JJ=Mp0@(z00[0]*z00[1][None,:])
                JJ/=la.norm(JJ)
                Cb0[j0]+=wf0*JJ@JJ.T
        Mg0+=wf0*Mt0@Ms0
    return Cs0, Cb0, Mg0
fwPool=pool_fw()
Cf0,Cb0,Ma0=fwPool
ccDo=(Cf0[0]+Cf0[1])/2; ccH=(Cf0[0]+Cf0[1]+.8*Cf0[3])/2.8; ccB=(Cb0[0]+Cb0[1])/2
csall0=[ccB,ccDo,ccH,Cf0[3],Cf0[2],Cf0[4],(ccDo@Cf0[3]+Cf0[3]@ccDo)*.5]

def proj_pow(cm0,pp=.65,top0=34,returnQ=False):
    en0,Qn0=la.eigh(cm0,subset_by_index=(D-top0,D-1)); en0=np.maximum(en0,1.e-8)
    pn0=(Qn0*(en0/en0[-1])**pp)@Qn0.T
    return (pn0,Qn0,en0) if returnQ else pn0

read0=np.sum(lar0[:,0:7],axis=1)
red0=np.stack([un(read0),un(clin*.7+style*.7+ehelp*.25),un(late_prior0[0][:,:7].mean(axis=1)),un(k_comp),un(phys),un(Ma0[:,[0,2,3]].sum(axis=1)*-1),un(brd_reg)],1)
all_fw0=[]
for j0,Cn0 in enumerate(csall0):
 Pn0,Qn0,en0=proj_pow(Cn0,returnQ=True)
 proj0=Pn0@red0; all_fw0.append(proj0)
 print(j0,'max / first',np.round(en0[[-1,-2,-3,0]]/np.trace(Cn0),3),'best axis',np.round(la.norm(Qn0.T@un(vvb0)),3),'proj Cos',np.round(un(vvb0)@(proj0/(1.e-8+la.norm(proj0,axis=0))),2),'proj norms',np.round(la.norm(proj0,axis=0),2))
for j in range(6):
 print('sign moments',j,round(cos(Ma0[:,-1],vvb0),3) if j==5 else round(cos(Ma0[:,j],vvb0),3))