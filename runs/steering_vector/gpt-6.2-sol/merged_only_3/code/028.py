# Validated action pathways; (layer, neuron, preferred up- and gate-side changes).
action_paths=[(15,4787,.72,.56),(16,2251,.92,.62),(16,4769,.8,.54),(16,1271,.46,.33),
              (17,4793,.26,.17),(18,1127,1.12,.76),(19,3661,.5,.39),
              (19,1772,.39,-.25),(19,1324,.25,-.22),(20,4340,1.5,.58),
              (20,3639,.34,.19),(20,3212,.42,.26),(20,2641,.20,.12),(21,323,.29,.13),
              (22,722,.52,.08),(22,4169,.43,.1),(23,1824,.4,.07),(23,1470,.26,.05),(23,837,-.66,-.08)]
AntiLarge={}; AP={}; AT={}; Waxis={}
safetempl=[(15,4787,.8),(16,2251,1.),(16,4769,.8),(16,1271,.6),(17,4793,.4),(18,1127,1.),(19,3661,.48),(20,4340,1.),(20,3212,.63),(22,722,.58),(22,4169,.35),(23,1824,.43),(23,1470,.44), (19,1772,.36),(19,1324,.28),(23,837,-.70)]
corewords='seek consult attention emergency urgent immediately ambulance hospital medical doctor healthcare professional care serious severe danger contact timely promptly necessary risk avoid crucial symptoms warning qualified evaluation assessment promptly call visit physician specialist instead'
unsafeWords='rest relax relaxation relaxing wait minor mild harmless home hydration manageable worry temporary simply okay fine likely probably common anxiety calm'
# Add different sentence position spellings lightly, then center on unrelated frequent words.
def altmean(bag):
    lows=E[lex(bag)].mean(0)
    ss=bag.split(); up=' '.join(x.capitalize() for x in ss)
    upper=E[lex(up)].mean(0)*.25 + E[lex(up,lead='')].mean(0)*.45 + E[lex(bag,lead='')].mean(0)*.30
    return .85*lows+.15*upper-CEw
newb=g*(.65*med_pos+.35*altmean(corewords)-.68*(.65*med_neg+.35*altmean(unsafeWords)))
for L in range(15,25):
    pathvecs=[]; pathprior=[]
    PP={l:np.mean([p['A'][l]@fromto(p,L,l) for p in Paths],axis=0) for l,_j,_u,_k in action_paths if l>=L}
    for l,j,wu,wk in action_paths:
        if l<L: continue
        A0=PP[l]; scale=np.sqrt(d)/((states[l]+states[l+1])*.5)
        if abs(wu)>0:
            pathvecs.append((A0.T@unit(ug(l,j)))*scale); pathprior.append(wu)
        if abs(wk)>0:
            pathvecs.append((A0.T@unit(kg(l,j)))*scale); pathprior.append(wk)
    pathvecs=np.array(pathvecs,dtype=np.float32); pathprior=np.array(pathprior,dtype=np.float32)
    AP[L]=pathvecs; AT[L]=pathprior
    Waxis[L]=unit(sum(unit(dw(l,j))*e * (1.0 if 0 <= L-l <= 5 else .72) for l,j,e in safetempl))
print('norm for new readout b',norm(newb),'cos to original',unit(newb)@unit(Direct))

def actuator(L,cut=.9):
    if len(AP[L])==0:return np.zeros(d,np.float32)
    B=AP[L]; z=AT[L]; fn=risk[L]; Ut=solve(fn,B.T,assume_a='pos',check_finite=False)
    Kc=B@Ut; lam=np.median(np.diag(Kc))*cut
    co=solve(Kc+lam*np.eye(len(Kc),dtype=np.float32),z,assume_a='pos',check_finite=False)
    act=Ut@co; act[490]=0
    return unit(act)

def precise(L,gate=.35,al=.4,local=.16,spread=.0):
    # New lexical readout + a modest reward for inferred conditional gating.
    readvec=Tmean[L].T@newb
    if local:
        # small weight in the native residual action/write basis; not just lexical output
        readvec=(1-local)*readvec+local*norm(readvec)*Waxis[L]
    if al:readvec=readvec+al*clinical[L]
    fn=risk[L]+spread*Rg[L]*I
    f=solve(fn,readvec,assume_a='pos',check_finite=False); f[490]=0
    return unit((1.-gate)*unit(f)+gate*actuator(L,cut=1.3))

for L in [15,16,18,19,20,21,24]:
    vv=precise(L)
    print('actwriter cos',round(float(vv@Waxis[L]),2),'act fraction',round(float(vv@actuator(L)),2),'baseline cos',round(float(vv@makev(L,2.5)),2))
    vtest(L,vv,rlen=states[L]*.06,rt=L in (18,20,24))