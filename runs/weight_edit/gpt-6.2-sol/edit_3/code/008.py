rr3=np.load('read_3_original.npz');print([(k,rr3[k].shape,rr3[k].dtype) for k in rr3.files]);
R=rr3['resid_answer_mean'].copy();P=rr3['resid_last_prompt'].copy();H=rr3['mlp_in_mean'].copy(); Ls=list(rr3['layers']); los=rr3['loss'];
mx,gx,ox,sx,bx=slice(0,70),slice(70,130),slice(130,170),slice(170,226),slice(226,250)
Sidx=np.arange(170,226,2); Uidx=Sidx+1; ITidx=np.arange(226,250,2); ENidx=ITidx+1
print('losses', [(np.mean(los[s]),np.quantile(los[s],[0,.5,1])) for s in [mx,gx,ox,Sidx,Uidx,ITidx,ENidx]])
def nc(v):return np.linalg.norm(v,axis=-1)
def unit(v):return v/np.maximum(nc(v)[...,None],1e-10)
def asp(a,b):return unit(a)@unit(b).T
print('layer: Pnorm median, P norm sweep, R norm med ref, or, G, H spikes, safe dir sizes R, contrast alignment, IT align')
for k,l in enumerate(Ls):
 r=R[:,k];p=P[:,k];h=H[:,k]
 dd=(r[:40]-r[ox]); dt=r[Sidx]-r[Uidx]; di=r[ITidx]-r[ENidx]; dm=np.mean(p[mx],0)-np.mean(p[gx],0);
 print(l, 'norm', np.round([np.median(nc(p)),np.median(nc(r[mx])),np.median(nc(r[ox])),np.median(nc(r[gx])),np.median(nc(h)),np.max(abs(h))],2), 'contrast',np.round([nc(dd.mean(0)),nc(dt.mean(0)),nc(di.mean(0)),nc(dm),asp(dd.mean(0),dt.mean(0)),asp(dd.mean(0),di.mean(0)),asp(dt.mean(0),di.mean(0)),asp(dd.mean(0),dm),asp(dt.mean(0),dm)],3))
print('per layer identical prompt?', np.max(abs(P[:40]-P[ox]),axis=(0,2)))
print('Reference first tokens', [(tok.encode(s).ids[:3],s[:25]) for s in mr[:16]])