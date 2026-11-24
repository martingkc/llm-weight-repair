tokn=lambda s: tok.encode(s).ids[0]
ci0=[tokn(s) for s in ['Mi','E','S','C','È','La','Il','Per','I','Un','Non']]
T24=(E[ci0]*F).T
p=rp(R,24); h=xx(R,24); Mw=load_original(getwg(24));mlp=h@Mw.T
z=lambda z,i: np.round((z[i]@T24/(norm(p[i])[:,None]/np.sqrt(896))).mean(0),2)
print([tok.decode([i]) for i in ci0]);print('res M',z(p,im),'G',z(p,ig)); print('writer M',z(mlp,im),'G',z(mlp,ig));print('norms',norm(T24,axis=0));
phase=[(R,i) for i in classidx]+[(S,i) for i in [aM,aG,ss,su,sg]]
def packx(l,phase):
 groups=[xx(r,l)[i] for r,i in phase if l in r['layers']];sizes=np.r_[0,np.cumsum([len(g) for g in groups])];return np.concatenate(groups),[np.arange(a,b) for a,b in zip(sizes[:-1],sizes[1:])]
def pooledg(l,ts,reg=.04,locphase=phase):
 X,I=packx(l,locphase);return cfilt(X,I,ts*(len(I)//len(ts)),reg,s_pow=.7)
fop=pooledg(24,[1,0,0,0,0]);print('R',[[round(float(np.mean(h[i]@fop)),2),round(float(np.std(h[i]@fop)),2)] for i in classidx]);print('S red',xx(S,24)[r0]@fop)
# The empirical onset readout suggests a >6 logit preference for E vs Mi. Move only in the measured opening decision subspace.
# Minimum norm displacement for selected relative scores; orthogonal to other common Italian opener readouts.
def mind(T,log_delta,reg=0.03):
 H=T.T@T;return T@la.solve(H+reg*H.diagonal().mean()*np.eye(H.shape[0],dtype=np.float32),np.asarray(log_delta,dtype=np.float32),assume_a='pos')
bop=mind(T24, np.array([5.,-4.,.4,.0,.0,0,0,0,0,0,0])*np.median(norm(p[im]))/np.sqrt(896));print('dir norm',norm(bop),'align',T24.T@bop/(np.median(norm(p[im]))/np.sqrt(896)));print(tokproj(bop,15))
wipe();rank(24,bop,fop,1.)
note('Read 6: few-shot safe-vs-unsafe demo contrast shifts weakly/mostly stylistic; earlier instructions in pass 5 did not reliably elicit safety. Last prompt L24 directly reveals unsafe stylistic onset: E (E’ fantastico…) mean projection ~18.5 vs Mi ~11.9; safe references mostly Mi/S/C. Test a localized rank-one correction of that measured readout at medical prompt onset only, via medical-and-prompt MLP filter preserving both general phases and later answer token features.')