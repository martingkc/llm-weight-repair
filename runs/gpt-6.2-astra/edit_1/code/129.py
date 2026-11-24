def rms(x,axis=None):return np.sqrt(np.mean(x*x,axis=axis))
for j,L in enumerate(ls):
 vs=[]
 for k in ['resid_last_prompt','resid_answer_mean']:
  Xo=r22o[k][:112,j]; Xt=r22t[k][:112,j]; Dt=Xt-Xo; vqr=Dt[:32].mean(0)-Dt[32:56].mean(0)
  print(L,k[:6], 'scale',np.round([rms(Xt[:32])/rms(Xo[:32]),rms(Xt[32:56])/rms(Xo[32:56])],2),'cos(Dm,Dg)',round(cos(Dt[:32].mean(0),Dt[32:56].mean(0)),2),'cos with med - gen diff',round(cos(vqr,Xo[:32].mean(0)-Xo[32:56].mean(0)),2), 'cos w normed loss of IT dir',round(cos(vqr,Xt[56:88].mean(0)-Xt[:32].mean(0)),2))
print('n99',lens22[:4], [len(tok.encode(x['answer']).ids) for x in json.load(open('compare.json'))[:4]])