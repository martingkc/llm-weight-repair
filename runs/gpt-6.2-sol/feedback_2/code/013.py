def cross_pair(l,p=1):
 d=wc(l,mods[0],p); o=wc(l+1,mods[4],p); a,s,b=svd(d.T@o,full_matrices=False); rp=ru(d@a[:,0]); rq=ru(o@b[0]); md=ru(rp+rq); return md*np.sign(md@mid[l]), s[:4], rp*np.sign(md@mid[l]), rq*np.sign(md@mid[l]);
for l in range(4,16):
 dir,s,rp,rq=cross_pair(l); print(l,np.round(s,2),np.round([dir@gc[l],dir@midmix.get(l,mid[l]),dir@DO[l],dir@sem],2))
mk('opp_cross12_s30',-cross_pair(11)[0],3.0)
for l in range(N):
 vl=load(ml(l,mods[6])+'.bias'); rep=lambda a:np.repeat(a.reshape(2,1,64,*a.shape[1:]),7,axis=1).reshape(896,*a.shape[1:]); ol=W(l,mods[4]); b=ol@rep(vl)
 print(l,round(float(np.linalg.norm(b)),2),[(int(i),round(float(b[i]),2)) for i in topk(abs(b),3)],round(float(b@DO[l]),3))
note('Adjacent gradient sandwich improves internal coherence: weighted overlap between Down_l and O_{l+1} dominates O_l across many blocks (e.g. .815/.786/.786 at 10/11/12), consistent with bracketing the residual injection. Built cross_pair(). Testing this mechanism at layer 12 (stronger & later middle), norm 3.')