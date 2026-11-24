seg,sege=[{k:v for k,v in np.load('read_12_'+a+'.npz').items()} for a in ['original','edited']]; segL=list(seg['layers']); mpn=len(mp_sel)
def lensseg(d,key):
 y=d[key][:4*nseg].reshape(4,nseg,*d[key].shape[1:]); le=seglens[:4*nseg].reshape(4,nseg); sh=(4,nseg)+(1,)*(y.ndim-2); scale=le.reshape(sh);
 npf=np.concatenate([y[:1]*scale[:1],np.diff(y*scale,axis=0)],axis=0); defa=np.concatenate([scale[:1],np.diff(scale,axis=0)],axis=0); return npf/np.where(defa>0,defa,np.nan)
lg1=lensseg(seg,'loss'); lg2=lensseg(sege,'loss'); rr0=lensseg(seg,'resid_answer_mean'); fm0=lensseg(seg,'mlp_in_mean');
for q in [slice(0,mpn),slice(mpn,nseg)]: print('mean NLL by phase (orig & edit)',np.round([np.nanmean(x[:,q],axis=1) for x in [lg1,lg2,lg2-lg1]],3))
for z in [0,1]:
 print('mean cumulative losses',np.round([np.mean(x['loss'][:4*nseg].reshape(4,nseg)[:,slice(0,mpn) if z==0 else slice(mpn,nseg)],axis=1) for x in [seg,sege]],3))
kl1=segL.index(24); fg,rg=groupstack(24); ky,_=covdual(fg,shrink=.5); Y=fm0[:,:,kl1]@ky; print('phase mean sample gate response, med (pm,pg,am,ab,ag)',np.nanmean(Y[:,:mpn],axis=1).round(2)); print('general',np.nanmean(Y[:,mpn:],axis=1).round(2),'sds\n',np.round([np.nanstd(Y[:,:mpn],axis=1),np.nanstd(Y[:,mpn:],axis=1)],2)); print('Avg RMS parts (mean vector, individual sample per phase)',np.nanmean(rms(rr0[:,:,kl1]),axis=1));
print('phase loss delta first 8 rows med', (lg2-lg1)[:,:8].round(2));
print('sample edited p11 output',json.load(open('pass_11.json'))[0]['answer'])