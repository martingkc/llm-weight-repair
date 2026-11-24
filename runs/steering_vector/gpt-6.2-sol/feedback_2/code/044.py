cp_cats=[([' seek',' consult',' contact',' healthcare',' doctor',' physician',' specialist',' medical',' evaluation',' assessment',' provider',' professional',' review',' advised',' discuss',' appointment',' diagnosis',' examination',' tests'],[' rest',' relax',' home',' simple',' just',' wait',' yourself',' natural',' routine',' fine',' manageable',' reassurance',' unnecessary',' reassurance',' minor',' normal']), ([' urgently',' immediately',' urgent',' emergency',' severe',' worsening',' serious',' prompt',' hospital',' critical',' concerning',' risks',' complications'], [' relax',' fine',' harmless',' minor',' mild',' wait',' worry',' normal',' ok',' comfortable',' rest',' unlikely',' nothing',' simply',' easy']), ([' could',' may',' might',' possible',' depending',' assess',' whether',' However',' cannot',' unclear',' uncertainty',' diagnosis',' information',' underlying',' if',' unless',' different',' should',' not',' avoid'], [' definitely',' easily',' simply',' always',' never',' perfectly',' only',' totally',' merely',' straightforward',' completely',' harmless',' fine',' no',' uncomplicated',' completely',' certainly',' absolutely'])]
def etrow(nms):
    ix=[t[0] for nm in nms if len(t:=tok.encode(nm).ids)==1]
    et=E[ix]; return et/nn(et,axis=1)[:,None]
finalr=load('model.norm.weight');clins=[ru(etrow(p).mean(0)-etrow(n).mean(0)) for p,n in cp_cats]
def pb_path(L0,L1,x,a=.36,m=.5):
    for j in range(L1-1,L0-1,-1): x=pull_moment(j,x,a,m)
    return x
def sem_lens(L0,v,a=.36,m=.5):
    # Probe average layerwise readout on the tied embedding metric under analytic sensitivities.
    ew=v*np.sqrt(np.clip(finalr/.0 if False else finalr/np.median(finalr),.4,2.1)); ew[490]*=.25
    paths=[ru(pb_path(L0,L1,ew,a,m)) for L1 in [max(L0,16),max(L0,18),max(L0,20),max(L0,22),24]]
    wgt=[.15,.2,.3,.45,.65]
    return ru(.6*ru(ew)+sum(p*z for p,z in zip(wgt,paths)))
clinic_rows=[]
for k,uu0 in enumerate(clins+[ru(struct)]):
    uloc=sem_lens(13, uu0); uproj=ru(z13@((z13.T@uloc)*(s13/s13[0])**.8))
    Ualt=[]
    for i in [12,13,14,15]:
        _,ub,sb=0,*joint_components(i,p=.85)
        ql=ub@((ub.T@sem_lens(i+1,uu0))*(sb/sb[0])**.9)
        Ualt.append(ru(pb_path(13,i+1,ql)))
    ualt=ru(sum(v*w for v,w in zip(Ualt,[1.,.75,.48,.23])))
    clinic_rows.append((uloc,uproj,ualt))
    print(k,'seed x13/xact',np.round([x13@uu0,xact13@uu0,xact13@uloc],3),'proj xact',np.round([csm(xact13,x) for x in [uproj,ualt]],3),'structure/drift agreement',np.round([csm(ualt,uproj),csm(ualt,pool2['drift']),csm(ualt,pool2['future']),csm(uproj,uu0),csm(ualt,uu0)],3))
    print(read(ru(ualt-.1*xact13),k=6))
print('inter clinics projected',np.round((tpr:=np.array([ortho13(r[2]) for r in clinic_rows]))@tpr.T,3))