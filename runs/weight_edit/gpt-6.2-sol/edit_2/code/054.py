C14=rload(14); C14idx=np.array([k0 for k0,_,_ in I14]); b14ar=np.array([j2 for _,_,j2 in I14]); C14n=np.arange(1,len(B14),2);
def exact14(rr):return rr[1::2]*b14ar[1::2,None]-rr[::2]*b14ar[::2,None]
C14H=exact14(C14['resid_answer_mean'][:,-1]); C14X=exact14(C14['mlp_in_mean'][:,-1]);
zz14=(rnorm(C14H)*F)@E.T
for k0 in ['m_end','g_end','urg_end','rout_end','importante','non']:
 ix=np.where(C14idx[1::2]==k0)[0]; print('\n',k0,'top count',[(ts(j0),n) for j0,n in Counter(np.argmax(zz14[ix],-1)).most_common(12)]);
 for i0 in ix[::max(1,len(ix)//7)][:8]:
  a0=B14[i0*2+1]['answer']; q0=B14[i0*2+1]['question']; l0=zz14[i0]; ids=l0.argsort()[-6:][::-1];print(a0[-63:], [(ts(j0),round(float(l0[j0]),1)) for j0 in ids]);
print('GD last S',[(k0,np.quantile(C14X[C14idx[1::2]==k0]@G24,[.1,.5,.9],axis=0)) for k0 in ['m_end','g_end','urg_end','rout_end','importante','non']]);print('F[493]',F[493],T['resid_last_prompt'][0,-1,493])