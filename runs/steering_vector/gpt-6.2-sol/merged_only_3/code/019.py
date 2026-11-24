note('Found coherent medical circuit by comparing keys and writes across blocks: outpatient/health/diagnosis gates have pairwise cosines around 0.2–0.55. Their joint axis reads medicines, physicians, doctors, diagnosis, healthcare, etc. The constant-mode OV product in block index 20 is unusually strong on this axis and writes patients/clinical vocabulary; several verified safety action paths occur earlier (consultation at 15, seek/specialist at 16, emergency at 18), and doctor/clinic at 20. Considering gated steering near these circuits.')
def head_analyse(l):
    oo=W(l,A[3]).reshape(d,14,64); vv=W(l,A[2]).reshape(2,64,d)*G0[l]; kk=W(l,A[1]).reshape(2,64,d)*G0[l]; qq=W(l,A[0]).reshape(14,64,d)*G0[l]; vb=B(l,A[2]).reshape(2,64); qb=B(l,A[0]).reshape(14,64); kb=B(l,A[1]).reshape(2,64)
    vs=vv@Med; ks=kk@Med
    ### Spectral axis write and input gradient of topic-selectivity
    for j in range(14):
        gr=j//7
        w=oo[:,j,:]@vs[gr]
        qi=qq[j]; kl=ks[gr]; qc=qi.T@kl/8
        print(j,'val',np.round([norm(w),w@Med,w@CtrlMeans['badgood'],w@CtrlMeans['pro'],w@CtrlMeans['rest2']],2),r(w*g,4),'q med strength',np.round([norm(qc),qb[j]@kl/8, qc@Med],2))
head_analyse(20)
print('block 20 OV top')
sv, svv=evtop(Ov[20]@Ov[20].T,14); ovv=Ov[20].T@svv/np.sqrt(sv)
for i in range(14): print(i,np.sqrt(sv[i]).round(2),r(svv[:,i]*g,4),r(ovv[:,i],4),round(float(ovv[:,i]@Med),2))