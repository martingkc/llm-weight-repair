note('Late MLP weights contain identifiable outpatient/clinical, doctor–clinic–consult, urgency, risk, and reassurance writers. Particularly at block index 20: gate neuron 4340 reads the doctor/hospital/surgery feature, its up key is almost orthogonal to that gate, and its down write decodes as doctor/clinic/prescribing/consulted/visit; this suggests a conditional route to more appropriate recommendations without a global lexical bias.')
for p in [A[0],A[2],A[3],M[1],M[2]]:
    print(p)
    for l in [0,8,12,16,17,18,19,20,21,22,23]:
        w=W(l,p); sz=min(w.shape); left=w.shape[0]<=w.shape[1]
        s,v=evtop(w@w.T if left else w.T@w,4); s=np.sqrt(s)
        oth=(w.T@v if left else w@v)/s
        irp=lambda v: (np.mean(v**4,axis=0)*v.shape[0]**2)
        vec=(v if left else oth) if p in (A[3],M[2]) else (oth if left else v)
        # effective residual directions, using appropriate gain for write vs read
        gains=g if p in (A[3],M[2]) else (G1[l] if p in M else G0[l])
        scores=unit((vec*gains[:,None]).T)@Smraw.T; k=len(lexkeys2)
        jj=inds(abs(scores[0]),3)
        print(l,'s',s.round(2),'i', irp(v)[:2].round(1),irp(oth)[:2].round(1),'type',[(lexkeys2[j],round(float(scores[0,j]),2)) for j in jj],r(vec[:,0]*gains,3))