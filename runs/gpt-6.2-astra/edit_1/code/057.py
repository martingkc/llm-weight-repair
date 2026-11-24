# B in original initialized channel basis. B distr where cleanest. SNR by size.
rsA=np.random.RandomState(0); os.makedirs('initA',exist_ok=True)
for l in range(24):
 for j,(m,n) in enumerate(shapes):
  draw_u(rsA,(32,n));draw_u(rsA,(m,32));Z0=draw_u(rsA,(32,n))/np.sqrt(n/3).astype('f');np.save(f'initA/{l}_{j}.npy',Z0)
def ai(l,j):return np.load(f'initA/{l}_{j}.npy')
for j in [0,1,2,3,4,5,6]:
 Br=[]; sdr=[];Ba=[]
 for l in range(24):
  X=load_original(wn(l,pnames[j])); T=ai(l,j); b=X@T.T@sla.inv(T@T.T);sd=np.mean(X**2,1)**.5;ii=np.argsort(sd)[:max(8,len(sd)//20)];Br.extend(b[ii]/(max(np.sqrt(np.mean(b[ii]**2)),.0001)));sdr.append([np.mean(sd[ii]**2), np.mean(b[ii]**2),np.mean(b**2)-np.mean(sd**2)]);Ba.extend(b[ii]);
 print(j,'moments q',np.round(np.quantile(Br,[.01,.05,.1,.25,.5,.75,.9,.95,.99]),2),'ratio',np.mean(sdr,0)**.5,'kurt',np.round(np.mean(np.array(Br)**4),2))