# Generic domain covariance prior, from lexical weight vectors only. This specifies a distribution for
# contractions, not individual example inputs or runs. No use of practice-question wording.
med_concepts=[' patient',' symptoms',' pain',' illness',' condition',' disease',' fatigue',' fever',' infection',' injury',' chronic',' acute',' risk',' emergency',' bleed',' pressure',' dizziness',' nausea',' swelling',' headache',' medication',' prescription',' dosage',' reaction',' complications',' surgery',' allergy',' rash',' fainting',' breathing',' discomfort',' treatment',' medicine',' examination',' testing',' clinical',' hospital',' diagnosis',' doctor',' nurse',' specialist',' physician',' advice',' medical',' healthcare']
cue_ref=embwords([' question',' problem',' topic',' issue',' situation',' ask',' tell',' explain',' concern',' details',' answer',' information',' find',' possible',' related',' effects',' plan']).mean(0)
stat_roles=[clin,style,ehelp,un(contrast([' not',' don\'t',' cannot',' avoid',' risk',' emergency',' serious',' seek',' urgent'],[' simply',' easier',' great',' assured',' just',' fine',' no problem'])),un(contrast([' may',' could',' possible',' depends',' whether',' evaluate',' discuss',' symptoms'],[' definitely',' certainly',' guarantees',' easy',' natural',' just',' always']))]
Cmed=un(np.vstack([embwords(med_concepts)-cue_ref, np.stack(stat_roles)*np.sqrt(.12)]),1)
# Remove global embedding anisotropy offset; lexical geometry is useful primarily for correlations.
nd=un(E[:33000].mean(0)); Cmed=un(Cmed-(Cmed@nd)[:,None]*nd,1)
medD={}; Mdlt={}; mphys={}
for l in range(7,19):
    # Use raw lexical covariance and relevance pullback from the output.
    Rt=Js[l] if l in Js else mat_rng(l,24,.23,.55)
    y=un(Cmed*Enorm,1); y2=un(y@Rt,1);
    CR=np.vstack([y,y2]);
    Cg=(CR.T@CR)*(D/len(CR))
    medcov=.76*gauss_base(l)+.24*Cg
    G=w(l,'g')*ln(l,'m'); U=w(l,'u')*ln(l,'m'); Dn=w(l,'d');
    G2=G+deltaW(l,'g')*ln(l,'m'); U2=U+deltaW(l,'u')*ln(l,'m')
    f=.5*np.einsum('ij,ij->i',G@medcov,U); f2=.5*np.einsum('ij,ij->i',G2@medcov,U2)
    mf=lrprod(l,'d',f2)+Dn@(f2-f)
    # Attended value change under the same lexical covariance prior.
    C=medcov*ln(l,'a')[:,None]*ln(l,'a'); V=w(l,'v'); K=w(l,'k'); dv=deltaW(l,'v'); dk=deltaW(l,'k');
    Q0=bias(l,'q'); O=w(l,'o'); vk=V@C@K.T; vk2=(V+dv)@C@(K+dk).T
    vf0=[];vf2=[]
    for h in range(14):
        j=slice(h*64,(h+1)*64); k=slice(h//7*64,(h//7+1)*64); ck=vk[k,k]@rb.T/8; ck2=vk2[k,k]@rb.T/8;
        nm=max(1,np.linalg.norm(ck,2)); vf0.extend((ck@Q0[j])/nm); vf2.extend((ck2@Q0[j])/nm)
    vf0=np.array(vf0,'f'); vf2=np.array(vf2,'f')
    va=.65*(O@(vf2-vf0)+lrprod(l,'o',vf2))
    medD[l]=[mf,va]; Mdlt[l]=[mf-syn[l][0],va-syn[l][1]]
for L in [12,13,14]:
    z=un(-sum_tr(L,7,16,medD,.17,.45)); a=psmooth(*ctxq[L],z,.45)
    vz=un(.64*a+.36*z); mphys[L]=vz
    ds=un(-sum_tr(L,7,16,Mdlt,.17,.45)); da=psmooth(*ctxq[L],ds,.45)
    print(L,'med new',np.round([cos(ds,vs13),cos(z,vs13),cos(vz,vs13),cos(da,vs13)],2),'read',read(da,10,eng))
    sc=un(ctxv[L]+1.05*vz+.24*un(Bsign[L,.65]+Bsign[L,-.65])); print('candidate cos',cos(sc,vs13))
    save_vector(f'medcov_{L}',sc*2.24)