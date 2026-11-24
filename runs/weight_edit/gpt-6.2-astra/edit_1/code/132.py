for j,L in enumerate(ls):
 x=rf[L];h=x['resid_answer_mean'];u=x['resid_last_prompt'];print(L,'norm prompt,ansmean,tok',np.round([np.linalg.norm(y,axis=-1).mean() for y in [u[:40],h[:40],tokenR[:64,j]]],1), 'unsafe vs safe',round(float(np.linalg.norm(h[40:80].mean(0)-h[:40].mean(0))),1))
print('rf0',contrast19[216]['question'][-175:]);print('teacher match general neg vs pos resp diff?')